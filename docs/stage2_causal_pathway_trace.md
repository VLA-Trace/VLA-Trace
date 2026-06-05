# Stage 2: Causal Pathway Trace

Stage 2 tests which modality pathways must remain open during action decoding.

Public core:

- `vla_trace.knockout.specs`: model-family-independent knockout specs and layer-window utilities.
- `vla_trace.knockout.builders`: numpy additive masks for OpenVLA and pi0.5 layouts.
- `vla_trace.knockout.sweeps`: layer/window expansion for sweeps.

Supported public families:

- `openvla`: causal layout `[BOS][IMAGE][TEXT][ACTION]`.
- `pi05`: bidirectional prefix `[IMAGE][TEXT]` plus action-generation suffix.

The public repository intentionally does not ship private cluster shell scripts.
Use YAML experiment configs and the `vla-trace knockout` command to describe
sweeps, then connect those specs to a model adapter for online evaluation.

`vla-trace knockout CONFIG --output mask.json` writes both the sweep metadata
and an additive-mask artifact for the configured public token layout. Full-size
public configs can produce tens to hundreds of MB of JSON; use smaller token
layouts for quick inspection. The artifact is meant for adapter integration and
offline inspection, not as a replacement for model-family-specific runtime hooks.

Knockout settings can be selected from the CLI with `--phase`, `--mode`,
`--text-scope`, `--layers`, `--center-layers`, and `--window-size`. Text scopes
separate semantic instruction tokens from public structural proxies such as BOS
and the final newline/suffix token; adapters may refine exact tokenizer offsets
before applying the mask online.
