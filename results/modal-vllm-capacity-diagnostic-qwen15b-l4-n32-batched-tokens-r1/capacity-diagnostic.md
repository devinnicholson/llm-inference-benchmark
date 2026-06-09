# vLLM Capacity Diagnostic

Model: `Qwen/Qwen2.5-1.5B-Instruct`
GPU: `L4`
Prompt profiles: `shared_prefix_mega_long_no_repeat_variant`
Warmup prompt profile: `neutral_mega_long`

| Requests | Prefix cache | Max model len | Max batched tokens | Source | Max seqs | GPU KV tokens | KV memory GiB | Max concurrency |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 32 | False | 3790 | 60640 | override | 32 | 158540 | 4.24 | 41.83 |
| 32 | False | 3790 | 121280 | override | 32 | 18854 | 0.5 | 4.97 |

## Reading

The higher-request-count shape raises configured max batched tokens by `2.000x` while reducing available GPU KV-cache tokens to `0.119x` of the baseline.
