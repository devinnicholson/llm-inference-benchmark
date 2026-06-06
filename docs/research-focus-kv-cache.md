# Research Focus: KV-Cache-Centered LLM Serving

## Position

The project should focus on KV cache because it is the clearest path from a
normal class project to a research-worthy ML systems artifact.

KV cache sits at the intersection of:

- model behavior: prompt length, output length, attention, long context
- GPU memory: bytes per token, fragmentation, paging, transfer, eviction
- scheduling: batching, prefill/decode interference, admission, fairness
- serving quality: TTFT, TPOT, p95/p99 latency, throughput, SLO misses
- systems design: placement, disaggregation, cache reuse, observability

That makes it a better research target than a generic "LLM benchmark" because
the experiments can answer narrow, falsifiable questions.

## Core Research Question

How do KV-cache memory pressure and scheduling policy interact under mixed
LLM inference workloads, and when do cache-aware policies improve tail latency
or throughput compared with simpler FIFO-style baselines?

## Candidate Subquestions

1. **Memory pressure**
   - How much KV cache does each workload class create?
   - At what prompt/output lengths does KV cache dominate available GPU memory?
   - How do long-context requests affect other users in a shared serving queue?

2. **Scheduling**
   - When does FIFO create head-of-line blocking due to long prefills?
   - Does shortest-prefill or shortest-remaining-cache-footprint scheduling help?
   - What does the policy sacrifice in fairness or starvation risk?

3. **Batching**
   - How do mixed prefill/decode batches affect TTFT and TPOT?
   - Can chunked prefill reduce decode stalls in the simulator?
   - Which workload distributions make batching look artificially good?

4. **Cache management**
   - What is the difference between contiguous allocation and paged allocation?
   - How much fragmentation can a naive allocator create?
   - When would eviction, CPU spill, or prefix reuse matter?

5. **Reproducibility**
   - Can another person rerun the workload and reproduce the same conclusion?
   - Are workload assumptions explicit enough to critique?
   - Are results reported as full distributions instead of single averages?

## Project Thesis

This repo should become a trace-driven KV-cache benchmark for LLM serving. The
artifact should include simple baselines, cache-aware policies, clear memory
models, latency distributions, and a written analysis of when each policy wins
or fails.

## Research Merit Boundary

This becomes research-worthy only if the final report contains:

- a clear question
- at least two baseline policies
- workload classes with documented assumptions
- KV-cache memory accounting
- latency and throughput distributions
- failure cases and negative results
- enough code/config detail for reproduction

Without those pieces, it is just a well-implemented class project.

## Initial Implementation Direction

The current Week 1 simulator estimates per-request KV-cache footprint using:

```text
bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_element
request_bytes = (prompt_tokens + output_tokens) * bytes_per_token
```

The default config approximates a 32-layer, 32-KV-head, 128-head-dim, FP16
decoder model. Model configs now live in `configs/models` so we can compare MHA,
GQA, MQA, and quantized KV-cache settings.

The simulator also builds an active KV-cache timeline with prompt allocation at
prefill start, decode-time KV growth, and request-level release at completion.

## Near-Term Milestones

1. Add active-memory accounting for concurrent requests.
2. Add GPU capacity configs and capacity-relative pressure metrics.
3. Implement FIFO, priority, shortest-prefill, and cache-aware schedulers.
4. Add workload generators for short chat, long-context RAG, coding, and batch.
5. Plot p95/p99 latency versus peak active KV-cache memory.
6. Write a short negative-results section for policies that look good only on
   unrealistic workloads.
