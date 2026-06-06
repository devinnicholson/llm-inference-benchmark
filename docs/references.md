# References

These are the starting primary sources for 568. We will add more as the project
narrows.

## LLM Serving and KV Cache

- vLLM / PagedAttention paper: https://arxiv.org/abs/2309.06180
  - Key idea for this repo: high-throughput LLM serving is strongly constrained
    by dynamic KV-cache memory, fragmentation, batching, and sequence length.

- Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving:
  https://arxiv.org/abs/2407.00079
  - Key idea for this repo: KV cache can be treated as a first-class systems
    resource that shapes disaggregation, transfer, scheduling, and placement.

- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large
  Language Model Serving: https://arxiv.org/abs/2401.09670
  - Key idea for this repo: prefill and decode have different latency targets
    and resource profiles, so separating them can reduce interference.

- Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked
  Prefills: https://arxiv.org/abs/2308.16369
  - Key idea for this repo: chunking long prefills can smooth interference with
    decode work and improve utilization.

## Structured LLM Programs

- SGLang paper: https://arxiv.org/abs/2312.07104
  - Key idea for this repo: structured and multi-step LLM programs need runtime
    support for parallelism, KV-cache reuse, and efficient constrained decoding.

## Production Inference Toolkit

- NVIDIA TensorRT-LLM docs: https://docs.nvidia.com/tensorrt-llm/
  - Key idea for this repo: production inference stacks expose features such as
    streaming, in-flight batching, paged attention, quantization, and multi-GPU
    support.

## GPU Kernel Programming

- Triton introduction: https://openai.com/index/triton/
  - Key idea for this repo: Triton provides a Python-like programming model for
    writing high-performance neural network kernels with block-level operations.
