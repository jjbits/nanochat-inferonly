#!/usr/bin/env python3
"""
Benchmark script for comparing inference performance.

Supports two modes:
- Default: benchmarks nanochat-inferonly (local code)
- --use-nanochat: benchmarks nanochat (imports from nanochat package)

Usage:
    python benchmark.py                    # benchmark nanochat-inferonly
    python benchmark.py --use-nanochat     # benchmark nanochat (FA3)
    python benchmark.py --num-tokens 256   # generate more tokens
"""

import argparse
import time
import sys
import os
import statistics
import torch

# Paths for weights
CHECKPOINT_DIR = "weights/weights_pankajmathur_huggingface_466/chatrl_checkpoints/d34"
STEP = 466
TOKENIZER_PATH = "weights/weights_karpathy_d34_huggingface/tokenizer.pkl"
META_PATH = f"{CHECKPOINT_DIR}/meta_000466.json"
MODEL_PATH = f"{CHECKPOINT_DIR}/model_000466.pt"


def benchmark_nanochat_inferonly(args):
    """Benchmark using local nanochat-inferonly code."""
    from loader import load_model_from_local
    from engine import Engine, KVCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading nanochat-inferonly model...")
    model, tokenizer, meta = load_model_from_local(
        model_path=MODEL_PATH,
        meta_path=META_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device=device
    )

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    return model, tokenizer, device, "nanochat-inferonly (SDPA)"


def benchmark_nanochat(args):
    """Benchmark using nanochat package (FA3)."""
    # Add nanochat to path
    nanochat_path = os.path.expanduser("~/projects/nanochat")
    if nanochat_path not in sys.path:
        sys.path.insert(0, nanochat_path)

    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine, KVCache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.abspath(CHECKPOINT_DIR)

    print(f"Loading nanochat model (FA3)...")
    model, tokenizer, meta = build_model(checkpoint_dir, STEP, device, phase="eval")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    return model, tokenizer, device, "nanochat (FA3)"


def run_benchmark(model, tokenizer, device, num_tokens, warmup_runs, num_runs, use_nanochat=False):
    """Run the actual benchmark."""
    from contextlib import nullcontext

    # Import the right Engine based on mode
    if use_nanochat:
        nanochat_path = os.path.expanduser("~/projects/nanochat")
        if nanochat_path not in sys.path:
            sys.path.insert(0, nanochat_path)
        from nanochat.engine import Engine
    else:
        from engine import Engine

    engine = Engine(model, tokenizer)

    # Set up autocast for bfloat16 on CUDA
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    # Build prompt tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    prompt = "What is the capital of France? Please explain in detail."
    prompt_tokens = [bos, user_start] + tokenizer.encode(prompt) + [user_end, assistant_start]

    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print(f"Max generate tokens: {num_tokens}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {num_runs}")

    prefill_times = []
    decode_times = []
    tokens_generated_list = []

    total_runs = warmup_runs + num_runs

    for run in range(total_runs):
        is_warmup = run < warmup_runs
        run_type = "warmup" if is_warmup else "benchmark"
        print(f"  Run {run + 1}/{total_runs} ({run_type})...", end=" ", flush=True)

        tokens_generated = 0
        generated_tokens = []

        # Time the full generation
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.perf_counter()
        prefill_done_time = None

        with autocast_ctx:
            for i, (token_column, token_masks) in enumerate(engine.generate(
                prompt_tokens,
                num_samples=1,
                max_tokens=num_tokens,
                temperature=0.0,  # Greedy for determinism
                top_k=None,
                seed=42
            )):
                if i == 0:
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    prefill_done_time = time.perf_counter()

                token = token_column[0]
                if token == assistant_end or token == bos:
                    break
                tokens_generated += 1
                generated_tokens.append(token)

        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.perf_counter()

        # Calculate times
        if prefill_done_time is None:
            prefill_done_time = start_time  # Edge case: no tokens generated

        prefill_time = prefill_done_time - start_time
        decode_time = end_time - prefill_done_time

        print(f"{tokens_generated} tokens, {(end_time - start_time)*1000:.1f}ms")

        if not is_warmup:
            prefill_times.append(prefill_time)
            decode_times.append(decode_time)
            tokens_generated_list.append(tokens_generated)

    # Calculate statistics
    num_prompt_tokens = len(prompt_tokens)
    avg_tokens_generated = statistics.mean(tokens_generated_list)

    prefill_tok_per_sec = [num_prompt_tokens / t if t > 0 else 0 for t in prefill_times]
    decode_tok_per_sec = [tg / t if t > 0 else 0 for tg, t in zip(tokens_generated_list, decode_times)]

    total_times = [pt + dt for pt, dt in zip(prefill_times, decode_times)]
    total_tokens = [num_prompt_tokens + tg for tg in tokens_generated_list]
    total_tok_per_sec = [tt / time if time > 0 else 0 for tt, time in zip(total_tokens, total_times)]

    return {
        "num_prompt_tokens": num_prompt_tokens,
        "avg_tokens_generated": avg_tokens_generated,
        "prefill_time_ms": statistics.mean(prefill_times) * 1000,
        "prefill_time_std_ms": statistics.stdev(prefill_times) * 1000 if len(prefill_times) > 1 else 0,
        "prefill_tok_per_sec": statistics.mean(prefill_tok_per_sec),
        "prefill_tok_per_sec_std": statistics.stdev(prefill_tok_per_sec) if len(prefill_tok_per_sec) > 1 else 0,
        "decode_time_ms": statistics.mean(decode_times) * 1000,
        "decode_time_std_ms": statistics.stdev(decode_times) * 1000 if len(decode_times) > 1 else 0,
        "decode_tok_per_sec": statistics.mean(decode_tok_per_sec),
        "decode_tok_per_sec_std": statistics.stdev(decode_tok_per_sec) if len(decode_tok_per_sec) > 1 else 0,
        "total_tok_per_sec": statistics.mean(total_tok_per_sec),
        "total_tok_per_sec_std": statistics.stdev(total_tok_per_sec) if len(total_tok_per_sec) > 1 else 0,
    }


def print_results(results, impl_name):
    """Print benchmark results."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS: {impl_name}")
    print(f"{'='*60}")
    print(f"Prompt tokens:     {results['num_prompt_tokens']}")
    print(f"Generated tokens:  {results['avg_tokens_generated']:.1f} (avg)")
    print()
    print(f"PREFILL:")
    print(f"  Time:            {results['prefill_time_ms']:.2f} ms (+/- {results['prefill_time_std_ms']:.2f})")
    print(f"  Throughput:      {results['prefill_tok_per_sec']:.1f} tok/s (+/- {results['prefill_tok_per_sec_std']:.1f})")
    print()
    print(f"DECODE:")
    print(f"  Time:            {results['decode_time_ms']:.2f} ms (+/- {results['decode_time_std_ms']:.2f})")
    print(f"  Throughput:      {results['decode_tok_per_sec']:.1f} tok/s (+/- {results['decode_tok_per_sec_std']:.1f})")
    print()
    print(f"TOTAL:")
    print(f"  Throughput:      {results['total_tok_per_sec']:.1f} tok/s (+/- {results['total_tok_per_sec_std']:.1f})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark nanochat inference')
    parser.add_argument('--use-nanochat', action='store_true',
                        help='Use nanochat package (FA3) instead of local inferonly code')
    parser.add_argument('--num-tokens', type=int, default=128,
                        help='Number of tokens to generate (default: 128)')
    parser.add_argument('--warmup-runs', type=int, default=2,
                        help='Number of warmup runs (default: 2)')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of benchmark runs (default: 5)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for model')
    args = parser.parse_args()

    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("WARNING: Running on CPU (will be slow)")

    # Load model based on mode
    if args.use_nanochat:
        model, tokenizer, device, impl_name = benchmark_nanochat(args)
    else:
        model, tokenizer, device, impl_name = benchmark_nanochat_inferonly(args)

    print(f"\nBenchmarking: {impl_name}")

    # Run benchmark
    results = run_benchmark(
        model, tokenizer, device,
        num_tokens=args.num_tokens,
        warmup_runs=args.warmup_runs,
        num_runs=args.num_runs,
        use_nanochat=args.use_nanochat
    )

    # Print results
    print_results(results, impl_name)


if __name__ == "__main__":
    main()
