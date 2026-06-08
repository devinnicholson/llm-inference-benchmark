# Prefix-Cache Prompt Token Audit

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
Profiles: `shared_prefix_long_no_repeat_variant,matched_unique_prefix_no_repeat_variant`
Shared profile: `shared_prefix_long_no_repeat_variant`
Control profile: `matched_unique_prefix_no_repeat_variant`
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
| Mean shared-minus-control reusable block fraction | 0.785 |
| Mean shared exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block tokens | 0.000 |
| Mean shared-minus-control exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block fraction | 0.000 |

## Scenario Audit

| Profile | Repeat | Requests | Common Prefix Tokens | Full Blocks | Reusable Block Tokens | Reusable Fraction | Unique Prompts | Duplicate Reusable Tokens | Duplicate Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_prefix_long_no_repeat_variant` | 0 | 16 | 303 | 18 | 4320 | 0.826 | 16 | 0 | 0.000 |
| `matched_unique_prefix_no_repeat_variant` | 0 | 16 | 24 | 1 | 240 | 0.045 | 16 | 0 | 0.000 |
| `shared_prefix_long_no_repeat_variant` | 1 | 16 | 302 | 18 | 4320 | 0.829 | 16 | 0 | 0.000 |
| `matched_unique_prefix_no_repeat_variant` | 1 | 16 | 24 | 1 | 240 | 0.045 | 16 | 0 | 0.000 |
| `shared_prefix_long_no_repeat_variant` | 2 | 16 | 319 | 19 | 4560 | 0.831 | 16 | 0 | 0.000 |
| `matched_unique_prefix_no_repeat_variant` | 2 | 16 | 24 | 1 | 240 | 0.043 | 16 | 0 | 0.000 |

## Shared Vs Control

| Repeat | Requests | Shared Blocks | Control Blocks | Block Delta | Reusable Token Delta | Shared Duplicate Tokens | Control Duplicate Tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 18 | 1 | 17 | 4080 | 0 | 0 |
| 1 | 16 | 18 | 1 | 17 | 4080 | 0 | 0 |
| 2 | 16 | 19 | 1 | 18 | 4320 | 0 | 0 |

Reusable block tokens estimate how many full leading-token blocks could be reused by requests after the first request in a batch. Exact-duplicate reusable tokens estimate repeated full-prompt token groups within the same scenario. This is a tokenizer/block audit, not a vLLM timing measurement.
