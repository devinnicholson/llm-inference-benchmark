# Prefix-Cache Batch-Pressure Comparison

Baseline: `n16 r8` from `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`
Candidate: `n32 r8` from `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

This compares repeat-count-matched Qwen 1.5B L4 mega-long artifacts while changing request count from n16 to n32.

## Key Differences

| Metric | n16 r8 | n32 r8 | n32 r8 - n16 r8 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Request count | 16 | 32 | +16 | Batch pressure is higher while model, GPU, prompt family, repeats, and output tokens stay matched. |
| Shared-minus-control direct counter delta | 92.628 pp | 95.797 pp | +3.169 pp | Direct measured-window KV reuse remains strong at the higher request count. |
| Shared direct counter hit rate | 93.075% | 96.243% | +3.168 pp | The shared-prefix workload still drives high direct cache reuse under n32. |
| Control direct counter hit rate | 0.447% | 0.446% | -0.001 pp | The matched unique-prefix control remains near zero in both artifacts. |
| Throughput-ratio delta | 7.491 | 9.863 | +2.372 | The n32 artifact shows a larger favorable throughput effect. |
| p95 first-event/TTFT ratio delta | -0.910 | -0.942 | -0.032 | First-token latency remains favorable and is slightly stronger under n32. |
| p95 latency-ratio delta | -0.892 | -0.911 | -0.019 | End-to-end p95 latency remains favorable and slightly stronger under n32. |
| p95 stream TPOT-ratio delta | -0.947 | -0.683 | +0.264 | Decode TPOT remains favorable, but the n32 improvement is weaker than n16. |

## Reading

The higher-request-count artifact preserves the direct KV-cache reuse claim and strengthens the throughput and first-token effects. The weaker stream TPOT delta means the next systems question is not whether prefix reuse exists, but why the n32 capacity profile changes decode behavior.

The observed n32 vLLM capacity drop came from Modal console logs, not from the persisted JSON schema, so this comparison should be paired with a follow-up scheduler/capacity diagnostic before making backend-level claims.
