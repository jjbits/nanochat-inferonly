"""
Inference engine with KV cache for efficient autoregressive generation.
"""

import torch
import torch.nn.functional as F


class KVCache:
    """
    KV Cache for flash_attn_with_kvcache API.
    Tensors are (B, T, H, D) layout. Cache is updated in-place.
    Works with both FA3 (Hopper+) and SDPA fallback via flash_attention module.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        self.cache_seqlens.zero_()

    def get_pos(self):
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)


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
        # Determine dtype based on device (cuda=bf16, else=fp32)
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
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
            "num_layers": m.n_layer,
            "device": device,
            "dtype": dtype,
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
