# nanochat-inferonly

Inference-only extraction from [karpathy/nanochat](https://github.com/karpathy/nanochat).

This project contains the minimal components needed to run inference with nanochat models, without the training code.

## Inference Speed Comparison

**Hardware:** NVIDIA GeForce RTX 4090, CUDA 12.8

**Model:** d34 RL-finetuned (2.2B params)

**Test Configuration:**
- Prompt tokens: 16
- Max generate tokens: 128 (actual: 80)
- Decoding: greedy (temperature=0)
- Warmup runs: 2
- Benchmark runs: 5

| Implementation | Prefill | Decode | Total |
|----------------|---------|--------|-------|
| nanochat-inferonly | 774.7 tok/s | 63.9 tok/s | 75.5 tok/s |
| nanochat | 779.1 tok/s | 65.3 tok/s | 77.0 tok/s |

Using PyTorch SDPA.

## License

MIT
