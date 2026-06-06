# Workload Generation

The repo includes deterministic synthetic workload generation for KV-cache and
scheduler experiments.

## Command

```bash
python3 scripts/generate_workload.py mixed_bursty \
  --requests 32 \
  --seed 568 \
  --output workloads/generated/mixed_bursty_32_seed568.json
```

## Profiles

| Profile | Purpose |
| --- | --- |
| `short_chat` | Many short prompts and short outputs for interactive chat pressure. |
| `long_rag` | Long prompts with moderate outputs for context-heavy retrieval workloads. |
| `coding` | Medium/long prompts with long outputs, closer to code assistant traffic. |
| `batch_summary` | Long prompts with lower-priority batch summarization behavior. |
| `mixed_bursty` | Bursty mixture of short chat, coding, long-context, and batch requests. |

The ranges are synthetic. They are not intended to reproduce a private
production trace. Their value is that the assumptions are explicit and
repeatable.

## Reproducibility Rule

Every generated workload should include the profile, request count, and seed in
the filename:

```text
workloads/generated/{profile}_{requests}_seed{seed}.json
```

That makes benchmark commands readable and avoids silently changing workloads
between runs.

