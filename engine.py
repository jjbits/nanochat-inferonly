"""
Inference engine with KV cache for efficient autoregressive generation.
"""

import torch
import torch.nn.functional as F


class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    Works with the GPT model to maintain cached keys/values across forward passes.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """Copy from another KV cache, optionally expanding batch dimension."""
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill from empty KV cache"

        self_layers, self_kv, self_batch, self_heads, self_seq, self_head_dim = self.kv_shape
        other_layers, other_kv, other_batch, other_heads, other_seq, other_head_dim = other.kv_shape

        assert self_layers == other_layers
        assert self_heads == other_heads
        assert self_head_dim == other_head_dim
        assert self_batch == other_batch or other_batch == 1
        assert self_seq >= other_seq

        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        """Insert new keys/values and return the full cache view."""
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)

        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add

        # Dynamically grow cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = ((t1 + 1024 + 1023) // 1024) * 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape

        self.kv_cache[layer_idx, 0, :, :, t0:t1, :] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1, :] = v

        key_view = self.kv_cache[layer_idx, 0, :, :, :t1, :]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1, :]

        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1

        return key_view, value_view


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample next token from logits of shape (B, vocab_size). Returns (B, 1)."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


class Engine:
    """High-level inference engine with KV caching."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=1.0, top_k=50, seed=42):
        """
        Generate tokens autoregressively with KV caching.

        Args:
            tokens: List of input token ids
            num_samples: Number of parallel samples to generate
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility

        Yields:
            (token_column, token_masks): Lists of generated tokens and masks per sample
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Special tokens for stopping
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Prefill with batch size 1
        m = self.model.config
        kv_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer
        }
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_kwargs)

        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)

        # Clone KV cache for each sample
        kv_length_hint = len(tokens) + max_tokens
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, **kv_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # Track completion per sample
        completed = [False] * num_samples

        for _ in range(max_tokens):
            if all(completed):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)
            token_column = next_ids[:, 0].tolist()
            token_masks = [1] * num_samples  # All sampled (no forced tokens in simple version)

            # Check for stop tokens
            for i, token in enumerate(token_column):
                if token == assistant_end or token == bos:
                    completed[i] = True

            yield token_column, token_masks

            # Prepare next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]

    def generate_text(self, prompt, max_tokens=256, temperature=1.0, top_k=50, seed=42):
        """
        Simple text generation from a prompt string.
        Returns the generated text (excluding prompt).
        """
        bos = self.tokenizer.get_bos_token_id()
        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        tokens = [bos, user_start] + self.tokenizer.encode(prompt) + [user_end, assistant_start]

        generated = []
        for token_col, _ in self.generate(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed
        ):
            token = token_col[0]
            if token == assistant_end or token == bos:
                break
            generated.append(token)

        return self.tokenizer.decode(generated)

    def generate_text_streaming(self, prompt, max_tokens=256, temperature=1.0, top_k=50, seed=42):
        """
        Streaming text generation - yields text chunks as they're generated.
        """
        bos = self.tokenizer.get_bos_token_id()
        user_start = self.tokenizer.encode_special("<|user_start|>")
        user_end = self.tokenizer.encode_special("<|user_end|>")
        assistant_start = self.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")

        tokens = [bos, user_start] + self.tokenizer.encode(prompt) + [user_end, assistant_start]

        for token_col, _ in self.generate(
            tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed
        ):
            token = token_col[0]
            if token == assistant_end or token == bos:
                break
            yield self.tokenizer.decode([token])
