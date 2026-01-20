# nanochat-inferonly

Inference-only extraction from [karpathy/nanochat](https://github.com/karpathy/nanochat).

This project contains the minimal components needed to run inference with nanochat models, without the training code.

## Inference Speed

**Hardware:** NVIDIA GeForce RTX 4090, CUDA 12.8

**Model:** d34 RL-finetuned (2.2B params)

**Test Configuration:**
- Prompt tokens: 16
- Generated tokens: 80
- Decoding: greedy (temperature=0)
- Runs: 5 (after 2 warmup)

| Metric | Throughput |
|--------|------------|
| Prefill | 681.5 tok/s |
| Decode | 57.4 tok/s |
| Total | 67.7 tok/s |

Using PyTorch SDPA (FA3 on Hopper+, SDPA fallback elsewhere).

## License

MIT
