# Configs

Configs are split by concern:

- `configs/models/*.yaml`: model family, adapter name, and token layout.
- `configs/benchmarks/*.yaml`: LIBERO suite metadata.
- `configs/experiments/*.yaml`: Stage 1/2 default experiment intent.
  Stage 3 commands are artifact-driven and take exported arrays/manifests
  directly from the CLI.

This keeps public configs free of private absolute paths. Users add checkpoint,
dataset, and output paths in local copies or through release artifacts.

Default public configs should keep resource paths as relative paths or `null`
placeholders. Runtime commands also accept user-local paths without requiring
those paths to be committed:

```bash
vla-trace cka --model OpenVLA --dataset libero_10 \
  --model-path checkpoints/openvla \
  --data-root datasets/LIBERO \
  --print-config

vla-trace knockout --model pi0.5 --dataset libero_goal \
  --model-config local/configs/pi05.yaml \
  --benchmark-config local/configs/libero_goal.yaml \
  --model-path checkpoints/pi05 \
  --data-root datasets/LIBERO \
  --print-config
```
