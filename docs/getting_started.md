# Getting Started

This repository exposes the Stage 1, Stage 2, and offline Stage 3 public
toolkit surface.

```bash
python -m pip install -e ".[test]"
vla-trace --help
vla-trace inspect-model configs/models/openvla.yaml
vla-trace knockout configs/experiments/openvla_libero_knockout.yaml
vla-trace attention-metrics --help
vla-trace patchmask --help
vla-trace input-edit --help
```

The README is the official command runbook for Stage 1 CKA, Stage 2 knockout,
and Stage 3 attention, PatchMask, input-edit, and visualization workflows.

Heavy model execution is adapter-backed. Public configs do not contain private
checkpoint paths, dataset roots, conda environments, or cluster GPU schedules.
Use saved representation banks for offline CKA, or provide model-specific
adapter dependencies for full extraction/evaluation.

Keep committed configs path-clean. Users can pass local resources at runtime:

```bash
vla-trace cka --model OpenVLA --dataset libero_10 \
  --model-path checkpoints/openvla \
  --data-root datasets/LIBERO \
  --print-config
```

Migration notes for old workspace-local scripts are also consolidated in the
README.
