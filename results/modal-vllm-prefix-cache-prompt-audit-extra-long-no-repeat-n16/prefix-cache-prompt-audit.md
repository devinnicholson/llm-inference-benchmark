# Prefix-Cache Prompt Token Audit

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
Profiles: `shared_prefix_extra_long_no_repeat_variant,matched_unique_prefix_extra_long_no_repeat_variant`
Shared profile: `shared_prefix_extra_long_no_repeat_variant`
Control profile: `matched_unique_prefix_extra_long_no_repeat_variant`
Request counts: `16`
Repeats: `1`
Scenario seed: `577`
KV cache block size: `16`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared common-prefix full blocks | 72.000 |
| Mean control common-prefix full blocks | 1.000 |
| Mean shared-minus-control common-prefix blocks | 71.000 |
| Mean shared-minus-control reusable block tokens | 17040.000 |
| Mean shared-minus-control reusable block fraction | 0.899 |
| Mean shared exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block tokens | 0.000 |
| Mean shared-minus-control exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block fraction | 0.000 |

## Scenario Audit

| Profile | Repeat | Requests | Common Prefix Tokens | Full Blocks | Reusable Block Tokens | Reusable Fraction | Unique Prompts | Duplicate Reusable Tokens | Duplicate Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_prefix_extra_long_no_repeat_variant` | 0 | 16 | 1161 | 72 | 17280 | 0.912 | 16 | 0 | 0.000 |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 0 | 16 | 24 | 1 | 240 | 0.013 | 16 | 0 | 0.000 |

## Shared Vs Control

| Repeat | Requests | Shared Blocks | Control Blocks | Block Delta | Reusable Token Delta | Shared Duplicate Tokens | Control Duplicate Tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 72 | 1 | 71 | 17040 | 0 | 0 |

Reusable block tokens estimate how many full leading-token blocks could be reused by requests after the first request in a batch. Exact-duplicate reusable tokens estimate repeated full-prompt token groups within the same scenario. This is a tokenizer/block audit, not a vLLM timing measurement.
