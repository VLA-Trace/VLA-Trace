"""Stage 3 behavior-trace utilities."""

from vla_trace.behavior.attention import (
    attention_entropy,
    attention_mass,
    compute_attention_metrics,
    compute_attention_metrics_from_files,
    iou_top_percent,
    peak_hit,
    write_plot_summary_csv,
)
from vla_trace.behavior.attention_views import extract_attention_views, parse_span, write_attention_view_artifact
from vla_trace.behavior.input_edit import build_input_edit_record, summarize_input_edit_results, write_input_edit_manifest
from vla_trace.behavior.patchmask import (
    ImageMaskEvalConfig,
    apply_image_mask_to_obs_inplace,
    apply_patch_mask,
    apply_patch_mask_to_views,
    build_patch_mask_from_maps,
    build_robot_target_masks,
    detect_instance_seg_keys,
    write_patchmask_artifact,
)

__all__ = [
    "ImageMaskEvalConfig",
    "apply_image_mask_to_obs_inplace",
    "apply_patch_mask",
    "apply_patch_mask_to_views",
    "attention_entropy",
    "attention_mass",
    "build_input_edit_record",
    "build_patch_mask_from_maps",
    "build_robot_target_masks",
    "compute_attention_metrics",
    "compute_attention_metrics_from_files",
    "detect_instance_seg_keys",
    "extract_attention_views",
    "iou_top_percent",
    "parse_span",
    "peak_hit",
    "summarize_input_edit_results",
    "write_input_edit_manifest",
    "write_attention_view_artifact",
    "write_patchmask_artifact",
    "write_plot_summary_csv",
]
