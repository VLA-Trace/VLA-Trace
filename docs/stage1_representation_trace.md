# Stage 1: Representation Trace

Stage 1 analyzes how VLA adaptation changes hidden-state geometry.

Public core:

- `vla_trace.representations.cka`: linear CKA, cross-modal CKA, and checkpoint-drift CKA.
- `vla_trace.representations.pooling`: token-layout pooling into
  `vision_pooled`, `text_pooled`, and `joint_pooled`.
- `vla_trace.representations.bank`: JSON/NPZ saved-bank helpers.
- `vla_trace.representations.drift`: config-facing report helpers used by the CLI.

Layer-wise checkpoint-drift reports include `matched_layer_summary` for the
paper-style view table over `vision_pooled`, `text_pooled`, and `joint_pooled`.

The stable input is a saved bank:

```text
{
  "metadata": {"sample_ids": [...]},
  "arrays": {
    "vision_pooled/layer_0": [[...], ...],
    "text_pooled/layer_0": [[...], ...],
    "joint_pooled/layer_0": [[...], ...]
  }
}
```

The loader also accepts the legacy extraction shape used by the research
scripts:

```text
{
  "sample_ids": [...],
  "representations": {
    "vision_pooled": {"0": [[...], ...]},
    "text_pooled": {"0": [[...], ...]},
    "joint_pooled": {"0": [[...], ...]}
  }
}
```

In that case the public loader flattens nested layers into keys such as
`vision_pooled/layer_0`.

Model-specific extraction for OpenVLA and pi0.5 should be implemented through
adapters so the CKA core remains usable without private runtime dependencies.
