# Prefix-Cache Prompt Token Audit

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
Profiles: `shared_prefix_long_variant,matched_unique_prefix_variant`
Request counts: `16`
Repeats: `3`
Scenario seed: `577`
KV cache block size: `16`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared common-prefix full blocks | 18.333 |
| Mean control common-prefix full blocks | 1.000 |
| Mean shared-minus-control common-prefix blocks | 17.333 |
| Mean shared-minus-control reusable block tokens | 4160.000 |
| Mean shared-minus-control reusable block fraction | 0.792 |
| Mean shared exact-duplicate reusable block tokens | 2602.667 |
| Mean control exact-duplicate reusable block tokens | 2602.667 |
| Mean shared-minus-control exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block fraction | 0.482 |

## Scenario Audit

| Profile | Repeat | Requests | Common Prefix Tokens | Full Blocks | Reusable Block Tokens | Reusable Fraction | Unique Prompts | Duplicate Reusable Tokens | Duplicate Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_prefix_long_variant` | 0 | 16 | 303 | 18 | 4320 | 0.834 | 8 | 2560 | 0.494 |
| `matched_unique_prefix_variant` | 0 | 16 | 24 | 1 | 240 | 0.045 | 8 | 2560 | 0.481 |
| `shared_prefix_long_variant` | 1 | 16 | 302 | 18 | 4320 | 0.836 | 8 | 2560 | 0.495 |
| `matched_unique_prefix_variant` | 1 | 16 | 24 | 1 | 240 | 0.045 | 8 | 2560 | 0.482 |
| `shared_prefix_long_variant` | 2 | 16 | 319 | 19 | 4560 | 0.839 | 8 | 2688 | 0.494 |
| `matched_unique_prefix_variant` | 2 | 16 | 24 | 1 | 240 | 0.043 | 8 | 2688 | 0.482 |

## Shared Vs Control

| Repeat | Requests | Shared Blocks | Control Blocks | Block Delta | Reusable Token Delta | Shared Duplicate Tokens | Control Duplicate Tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 18 | 1 | 17 | 4080 | 2560 | 2560 |
| 1 | 16 | 18 | 1 | 17 | 4080 | 2560 | 2560 |
| 2 | 16 | 19 | 1 | 18 | 4320 | 2688 | 2688 |

Reusable block tokens estimate how many full leading-token blocks could be reused by requests after the first request in a batch. Exact-duplicate reusable tokens estimate repeated full-prompt token groups within the same scenario. This is a tokenizer/block audit, not a vLLM timing measurement.
