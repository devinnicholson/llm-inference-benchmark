# Single-L4 Feasibility Failure at a 60,640-Token Scheduler Budget

This attempt did not produce request-timing results. It failed during vLLM
engine profiling and KV-cache initialization for
`Qwen/Qwen2.5-7B-Instruct` on one NVIDIA L4.

Observed diagnostics:

- Effective maximum model length: 3,628 tokens.
- Configured `max_num_batched_tokens`: 60,640.
- vLLM's 16-sequence ceiling at that length: 58,048 tokens.
- Model-weight memory reported during loading: 14.29 GiB.
- Failed allocation: 2.14 GiB with 1.97 GiB reported free.

Because the engine never initialized, this run is evidence about feasibility,
not latency or throughput. The matched TP1-versus-TP2 experiment reduced the
scheduler budget to 8,192 tokens, within the initialization envelope of both
topologies and sufficient to execute the fixed workload with chunked prefill.
