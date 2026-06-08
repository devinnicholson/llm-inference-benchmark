# Prefix-Cache Prompt Token Audit

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
Profiles: `shared_prefix_long_variant,matched_unique_prefix_variant`
Request counts: `2,4,8`
Repeats: `3`
Scenario seed: `577`
KV cache block size: `16`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared common-prefix full blocks | 18.333 |
| Mean control common-prefix full blocks | 1.000 |
| Mean shared-minus-control common-prefix blocks | 17.333 |
| Mean shared-minus-control reusable block tokens | 1016.889 |
| Mean shared-minus-control reusable block fraction | 0.598 |

## Scenario Audit

| Profile | Repeat | Requests | Common Prefix Tokens | Full Blocks | Reusable Block Tokens | Reusable Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_prefix_long_variant` | 0 | 2 | 303 | 18 | 288 | 0.444 |
| `shared_prefix_long_variant` | 0 | 4 | 303 | 18 | 864 | 0.666 |
| `shared_prefix_long_variant` | 0 | 8 | 303 | 18 | 2016 | 0.778 |
| `matched_unique_prefix_variant` | 0 | 2 | 24 | 1 | 16 | 0.024 |
| `matched_unique_prefix_variant` | 0 | 4 | 24 | 1 | 48 | 0.036 |
| `matched_unique_prefix_variant` | 0 | 8 | 24 | 1 | 112 | 0.042 |
| `shared_prefix_long_variant` | 1 | 2 | 302 | 18 | 288 | 0.444 |
| `shared_prefix_long_variant` | 1 | 4 | 302 | 18 | 864 | 0.668 |
| `shared_prefix_long_variant` | 1 | 8 | 302 | 18 | 2016 | 0.780 |
| `matched_unique_prefix_variant` | 1 | 2 | 24 | 1 | 16 | 0.024 |
| `matched_unique_prefix_variant` | 1 | 4 | 24 | 1 | 48 | 0.036 |
| `matched_unique_prefix_variant` | 1 | 8 | 24 | 1 | 112 | 0.042 |
| `shared_prefix_long_variant` | 2 | 2 | 319 | 19 | 304 | 0.446 |
| `shared_prefix_long_variant` | 2 | 4 | 319 | 19 | 912 | 0.670 |
| `shared_prefix_long_variant` | 2 | 8 | 319 | 19 | 2128 | 0.783 |
| `matched_unique_prefix_variant` | 2 | 2 | 24 | 1 | 16 | 0.023 |
| `matched_unique_prefix_variant` | 2 | 4 | 24 | 1 | 48 | 0.034 |
| `matched_unique_prefix_variant` | 2 | 8 | 24 | 1 | 112 | 0.040 |

## Shared Vs Control

| Repeat | Requests | Shared Blocks | Control Blocks | Block Delta | Reusable Token Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 18 | 1 | 17 | 272 |
| 0 | 4 | 18 | 1 | 17 | 816 |
| 0 | 8 | 18 | 1 | 17 | 1904 |
| 1 | 2 | 18 | 1 | 17 | 272 |
| 1 | 4 | 18 | 1 | 17 | 816 |
| 1 | 8 | 18 | 1 | 17 | 1904 |
| 2 | 2 | 19 | 1 | 18 | 288 |
| 2 | 4 | 19 | 1 | 18 | 864 |
| 2 | 8 | 19 | 1 | 18 | 2016 |

Reusable block tokens estimate how many full leading-token blocks could be reused by requests after the first request in a batch. This is a tokenizer/block audit, not a vLLM timing measurement.
