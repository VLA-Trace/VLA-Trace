"""Command line interface for the public VLA-Trace toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_trace import __version__
from vla_trace.adapters import load_model_spec
from vla_trace.io import load_config, write_json
from vla_trace.io.registry import (
    DATASET_CONFIGS,
    DEFAULT_CENTER_LAYERS,
    DEFAULT_KNOCKOUT_MODE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TOKEN_LAYOUTS,
    DEFAULT_WINDOW_SIZE,
    MODEL_CONFIGS,
    dataset_config_path,
    model_config_path,
    normalize_dataset,
    normalize_model,
)


MODEL_HELP = "Model shortcut: OpenVLA or pi0.5"
DATASET_HELP = "LIBERO suite: libero_10, libero_goal, libero_object, or libero_spatial"


def _cmd_inspect_model(args: argparse.Namespace) -> int:
    spec = load_model_spec(args.config)
    print(f"name: {spec.name}")
    print(f"family: {spec.family}")
    print(f"adapter: {spec.adapter}")
    print(f"checkpoint: {spec.checkpoint or '<set by user>'}")
    print(f"visual_tokens: {spec.token_layout.visual_tokens}")
    print(f"action_tokens: {spec.token_layout.action_tokens}")
    print(f"structural_tokens: {', '.join(spec.token_layout.structural_tokens or [])}")
    print(f"supported_stages: {', '.join(spec.supported_stages)}")
    print(f"supports_stage3: {spec.supports_stage3}")
    if spec.notes:
        print(f"notes: {spec.notes}")
    return 0


def _cmd_collect_repr(args: argparse.Namespace) -> int:
    from vla_trace.representations.extraction import collect_representation_bank_from_config

    cfg = load_config(args.config) if args.config else {}
    if args.manifest:
        cfg["manifest_path"] = args.manifest
    if args.hidden_state_dir:
        cfg["hidden_state_dir"] = args.hidden_state_dir
    if args.adapter:
        cfg["adapter"] = args.adapter
    if args.bank_output:
        cfg["output_path"] = args.bank_output
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.checkpoint_name:
        cfg["checkpoint_name"] = args.checkpoint_name
    if args.token_group:
        cfg["token_groups"] = _parse_token_group_pairs(args.token_group)
    if args.model:
        model = normalize_model(args.model)
        cfg["model"] = model_config_path(model)
        cfg["family"] = model
        cfg.setdefault("token_layout", dict(DEFAULT_TOKEN_LAYOUTS[model]))
    if args.dataset:
        dataset = normalize_dataset(args.dataset)
        cfg["benchmark"] = dataset_config_path(dataset)
    _apply_user_path_overrides(cfg, args)
    if cfg.get("manifest_path") and cfg.get("output_path"):
        result = collect_representation_bank_from_config(cfg)
        if args.output:
            write_json(args.output, {"command": "collect-repr", "config": cfg, **result})
        print(result["output_path"])
        return 0
    plan = {
        "command": "collect-repr",
        "status": "needs-manifest-and-output",
        "config": cfg,
        "message": (
            "Provide --manifest and --bank-output. Each manifest row may point "
            "to hidden_states_path, or pass --hidden-state-dir / --adapter."
        ),
    }
    if args.output:
        write_json(args.output, plan)
    else:
        print(plan["message"])
    return 0


def _cmd_export_libero_manifest(args: argparse.Namespace) -> int:
    from vla_trace.representations.manifest import export_libero_manifest

    result = export_libero_manifest(
        data_root=args.data_root,
        data_mix=args.data_mix,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        image_size=args.image_size,
        train=not args.no_train,
        shuffle_buffer_size=args.shuffle_buffer_size,
        per_task_quota=args.per_task_quota,
        max_stream_reads=args.max_stream_reads,
    )
    _print_json(result)
    return 0


def _cmd_convert_bank(args: argparse.Namespace) -> int:
    from vla_trace.representations.extraction import convert_bank

    output = convert_bank(args.input, args.output, metadata=_parse_key_value_pairs(args.metadata or []))
    print(str(output))
    return 0


def _cmd_cka(args: argparse.Namespace) -> int:
    from vla_trace.representations.drift import analyze_cross_modal_from_config, analyze_drift_from_config

    cfg = _resolve_cka_config(args)
    if args.print_config:
        _print_json(cfg)
        return 0
    analysis = cfg.get("analysis", "cross_modal")
    if analysis == "cross_modal":
        result = analyze_cross_modal_from_config(cfg)
    elif analysis in {"checkpoint_drift", "drift"}:
        result = analyze_drift_from_config(cfg)
    else:
        raise ValueError(f"Unsupported CKA analysis: {analysis}")
    print(result)
    return 0


def _cmd_knockout(args: argparse.Namespace) -> int:
    from vla_trace.knockout.builders import build_additive_mask, make_openvla_partitions, make_pi05_partitions
    from vla_trace.knockout.sweeps import build_sweep_from_config

    cfg = _resolve_knockout_config(args)
    if args.print_config:
        _print_json(cfg)
        return 0
    sweep = build_sweep_from_config(cfg)
    if args.output:
        payload = _strict_json_payload(sweep.to_dict())
        payload["config"] = _strict_json_payload(cfg)
        token_layout = sweep.token_layout
        if sweep.spec.family == "openvla":
            partitions = make_openvla_partitions(
                visual_tokens=token_layout["visual_tokens"],
                text_tokens=token_layout["text_tokens"],
                action_tokens=token_layout["action_tokens"],
            )
        else:
            partitions = make_pi05_partitions(
                visual_tokens=token_layout["visual_tokens"],
                text_tokens=token_layout["text_tokens"],
                action_tokens=token_layout["action_tokens"],
                token_order=str(token_layout.get("token_order", "text,visual,action")),
            )
        mask = build_additive_mask(sweep.spec, partitions)
        artifact_values, artifact_block_value = _strict_mask_values(mask.values)
        payload["mask"] = {
            "shape": list(mask.values.shape),
            "layers": list(mask.layers),
            "block_value": artifact_block_value,
            "block_value_semantics": "additive attention block; converted from internal -inf for strict JSON",
            "values": artifact_values,
        }
        if sweep.phase_specs:
            phase_masks = {}
            for phase, phase_spec in sweep.phase_specs.items():
                phase_mask = build_additive_mask(phase_spec, partitions)
                phase_values, phase_block_value = _strict_mask_values(phase_mask.values)
                phase_masks[phase] = {
                    "shape": list(phase_mask.values.shape),
                    "layers": list(phase_mask.layers),
                    "block_value": phase_block_value,
                    "block_value_semantics": "phase-specific additive attention block",
                    "values": phase_values,
                }
            payload["phase_masks"] = phase_masks
        write_json(args.output, payload)
    print(f"{sweep.summary()} model={_display_model(cfg)} dataset={_display_dataset(cfg)}")
    return 0


def _cmd_knockout_sweep(args: argparse.Namespace) -> int:
    from vla_trace.knockout.presets import build_standard_knockout_manifest

    manifest = build_standard_knockout_manifest(
        model=args.model,
        dataset=args.dataset,
        output_path=args.output,
        window_size=args.window_size,
        num_layers=args.num_layers,
        trials=args.trials,
    )
    print(str(args.output) if args.output else f"jobs={manifest['n_jobs']}")
    return 0


def _cmd_eval_libero(args: argparse.Namespace) -> int:
    from vla_trace.evaluation.libero import (
        build_libero_eval_plan,
        resolve_knockout_config_for_eval,
        run_libero_evaluation,
        setting_name_from_knockout,
    )

    cfg = load_config(args.config) if args.config else {}
    model = _infer_model_shortcut(args.model or cfg.get("family") or cfg.get("model", "OpenVLA"))
    dataset = _infer_dataset_shortcut(
        args.dataset or cfg.get("dataset") or cfg.get("suite") or cfg.get("benchmark", "libero_10")
    )
    output_dir = args.output_dir or cfg.get("output_dir") or f"runs/{model}_{dataset}_eval"

    if args.knockout_manifest:
        jobs = _select_knockout_manifest_jobs(
            args.knockout_manifest,
            job_index=args.job_index,
            job_tag=args.job_tag,
            max_jobs=args.max_jobs,
        )
        results = []
        for idx, job in enumerate(jobs):
            knockout_config = _knockout_config_from_manifest_job(model, job, resolve_knockout_config_for_eval)
            setting = str(job.get("setting") or job.get("setting_suffix") or setting_name_from_knockout(knockout_config))
            tag = str(job.get("tag") or f"job_{idx:04d}_{setting}")
            request = _build_eval_request(
                args,
                cfg,
                model=model,
                dataset=dataset,
                output_dir=str(Path(output_dir) / tag),
                knockout_config=knockout_config,
                setting=setting,
                result_name=args.result_name,
            )
            results.append(build_libero_eval_plan(request) if args.print_plan else run_libero_evaluation(request))
        payload = {
            "status": "planned" if args.print_plan else "ok",
            "command": "eval-libero",
            "manifest": args.knockout_manifest,
            "n_jobs": len(jobs),
            "results": results,
        }
        if args.print_plan:
            _print_json(payload)
        elif args.output:
            write_json(args.output, payload)
            print(str(args.output))
        else:
            for result in results:
                print(str(result["result_path"]))
        return 0

    knockout_layers = _resolve_eval_knockout_layers(args, cfg)
    knockout_config = resolve_knockout_config_for_eval(
        model=model,
        phase=args.phase or str(cfg.get("phase", "generation")),
        mode=args.mode or str(cfg.get("mode", "baseline")),
        layers=knockout_layers,
        direction=args.direction or cfg.get("direction"),
        text_scope=args.text_scope or str(cfg.get("text_scope", "all")),
        prefill_mode=args.prefill_mode or cfg.get("prefill_mode"),
        generation_mode=args.generation_mode or cfg.get("generation_mode"),
    )
    setting = args.setting or str(cfg.get("setting") or setting_name_from_knockout(knockout_config))
    if isinstance(knockout_config, dict):
        if args.center_layer is not None:
            knockout_config["center_layer"] = args.center_layer
        if args.window_size is not None:
            knockout_config["window_size"] = args.window_size
    request = _build_eval_request(
        args,
        cfg,
        model=model,
        dataset=dataset,
        output_dir=str(output_dir),
        knockout_config=knockout_config,
        setting=setting,
        result_name=args.result_name or cfg.get("result_name"),
    )
    if args.print_plan:
        _print_json(build_libero_eval_plan(request))
        return 0
    result = run_libero_evaluation(request)
    if args.output:
        write_json(args.output, result)
        print(str(args.output))
    else:
        print(str(result["result_path"]))
    return 0


def _build_eval_request(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    *,
    model: str,
    dataset: str,
    output_dir: str,
    knockout_config: dict[str, Any] | None,
    setting: str,
    result_name: str | None,
) -> Any:
    from vla_trace.evaluation.libero import LiberoEvalRequest

    return LiberoEvalRequest(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        model_path=args.model_path or cfg.get("model_path"),
        data_root=args.data_root or cfg.get("data_root"),
        model_config=args.model_config or cfg.get("model_config"),
        benchmark_config=args.benchmark_config or cfg.get("benchmark_config"),
        config_path=args.config_path or cfg.get("config_path"),
        adapter_factory=args.adapter_factory or cfg.get("adapter_factory"),
        vlm4vla_root=args.vlm4vla_root or cfg.get("vlm4vla_root"),
        openpi_root=args.openpi_root or cfg.get("openpi_root"),
        openpi_config_name=args.openpi_config_name or cfg.get("openpi_config_name"),
        tokenizer_path=args.tokenizer_path or cfg.get("tokenizer_path"),
        device=args.device or str(cfg.get("device", "cuda")),
        seed=args.seed if args.seed is not None else int(cfg.get("seed", 0)),
        task_ids=_parse_task_ids(args.task_ids if args.task_ids is not None else cfg.get("task_ids")),
        num_trials_per_task=args.num_trials_per_task
        if args.num_trials_per_task is not None
        else int(cfg.get("num_trials_per_task", 20)),
        max_steps=args.max_steps if args.max_steps is not None else cfg.get("max_steps"),
        num_steps_wait=args.num_steps_wait
        if args.num_steps_wait is not None
        else int(cfg.get("num_steps_wait", 10)),
        execute_step=args.execute_step if args.execute_step is not None else int(cfg.get("execute_step", 1)),
        replan_steps=args.replan_steps if args.replan_steps is not None else cfg.get("replan_steps"),
        center_crop=bool(args.center_crop or cfg.get("center_crop", False)),
        save_video=bool(args.save_video or cfg.get("save_video", False)),
        video_every=args.video_every if args.video_every is not None else int(cfg.get("video_every", 0)),
        use_openvla_prompt=bool(args.use_openvla_prompt or cfg.get("use_openvla_prompt", False)),
        single_unnorm=bool(args.single_unnorm or cfg.get("single_unnorm", False)),
        knockout_config=knockout_config,
        setting=setting,
        result_name=result_name,
        mock_env=bool(args.mock_env or cfg.get("mock_env", False)),
        dry_run=bool(args.dry_run or cfg.get("dry_run", False)),
        extra_config=dict(cfg.get("extra_config", {})),
    )


def _cmd_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out = Path(args.output or cfg.get("output_path", "runs/report.json"))
    write_json(out, {"status": "ok", "config": cfg})
    print(str(out))
    return 0


def _cmd_plot_cka(args: argparse.Namespace) -> int:
    from vla_trace.visualization import plot_cka_report

    output = plot_cka_report(
        args.report,
        args.output,
        title=args.title,
        source_data=args.source_data,
    )
    print(str(output))
    return 0


def _cmd_plot_knockout(args: argparse.Namespace) -> int:
    from vla_trace.visualization import plot_knockout_results

    output = plot_knockout_results(
        args.inputs,
        args.output,
        model=args.model,
        dataset=args.dataset,
        setting=args.setting,
        title=args.title,
        source_data=args.source_data,
    )
    print(str(output))
    return 0


def _cmd_plot_knockout_line(args: argparse.Namespace) -> int:
    from vla_trace.visualization.knockout import plot_knockout_line_grid

    outputs = plot_knockout_line_grid(
        args.inputs or None,
        args.output_dir,
        selected_csv=args.selected_csv,
        baseline_csv=args.baseline_csv,
        model=args.model,
        dataset=args.dataset,
        protocol=args.protocol,
        source_data_dir=args.source_data_dir,
    )
    for output in outputs:
        print(str(output))
    return 0


def _cmd_plot_cka_publication(args: argparse.Namespace) -> int:
    from vla_trace.visualization.cka_publication import plot_cka_publication

    kwargs: dict[str, Any] = {}
    if args.datasets:
        kwargs["datasets"] = tuple(_parse_str_csv(args.datasets))
    if args.models:
        kwargs["models"] = tuple(normalize_model(model) for model in _parse_str_csv(args.models))
    outputs = plot_cka_publication(
        _parse_key_value_pairs(args.report or []),
        args.output_dir,
        **kwargs,
    )
    for output in outputs:
        print(str(output))
    return 0


def _cmd_plot_attention(args: argparse.Namespace) -> int:
    from vla_trace.visualization import plot_attention_iou

    output = plot_attention_iou(
        args.csv,
        args.output,
        metric=args.metric,
        title=args.title,
    )
    print(str(output))
    return 0


def _cmd_plot_attention_map(args: argparse.Namespace) -> int:
    from vla_trace.visualization import plot_attention_map

    output = plot_attention_map(
        args.array,
        args.output,
        key=args.key,
        keep_last_dims=args.keep_last_dims,
        normalize=args.normalize,
        kind=args.kind,
        x_labels=_parse_str_csv(args.x_labels) if args.x_labels else None,
        y_labels=_parse_str_csv(args.y_labels) if args.y_labels else None,
        title=args.title,
    )
    print(str(output))
    return 0


def _cmd_attention_metrics(args: argparse.Namespace) -> int:
    from vla_trace.behavior import compute_attention_metrics_from_files, write_plot_summary_csv

    rows = compute_attention_metrics_from_files(
        args.attention,
        args.masks,
        args.output,
        metadata_path=args.metadata,
        objects_path=args.objects,
        summary_path=args.summary,
        grid_shape=(args.grid_size, args.grid_size),
        top_percent=args.top_percent,
        fixed_threshold=args.fixed_threshold,
    )
    if args.plot_summary:
        write_plot_summary_csv(args.plot_summary, rows)
        print(f"{args.output} rows={len(rows)} plot_summary={args.plot_summary}")
    else:
        print(f"{args.output} rows={len(rows)}")
    return 0


def _cmd_attention_export(args: argparse.Namespace) -> int:
    from vla_trace.behavior.attention_views import parse_span, write_attention_view_artifact

    output = write_attention_view_artifact(
        args.attention,
        args.output,
        key=args.key,
        visual_span=parse_span(args.visual_span),
        text_span=parse_span(args.text_span),
        action_span=parse_span(args.action_span),
        normalize_rows=not args.no_normalize_rows,
    )
    print(str(output))
    return 0


def _cmd_attention_overlay(args: argparse.Namespace) -> int:
    from vla_trace.behavior.visualization import plot_attention_overlay

    output = plot_attention_overlay(
        args.image,
        args.attention,
        args.masks,
        args.output,
        attention_key=args.attention_key,
        mask_key=args.mask_key,
        grid_shape=(args.grid_size, args.grid_size),
        top_percent=args.top_percent,
        alpha=args.alpha,
    )
    print(str(output))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from vla_trace.doctor import run_doctor

    model_config = args.model_config
    if args.model and not model_config:
        model_config = model_config_path(normalize_model(args.model))
    benchmark_config = args.benchmark_config
    if args.dataset and not benchmark_config:
        benchmark_config = dataset_config_path(normalize_dataset(args.dataset))
    report = run_doctor(
        model_config=model_config,
        benchmark_config=benchmark_config,
        model_path=args.model_path,
        data_root=args.data_root,
        banks=_parse_key_value_pairs(args.bank or []),
        attention_path=args.attention,
        masks_path=args.masks,
        results_path=args.results,
        input_edits_path=args.input_edits,
    )
    if args.output:
        write_json(args.output, report)
        print(str(args.output))
    else:
        _print_json(report)
    return 1 if report["status"] == "error" and args.strict else 0


def _cmd_patchmask(args: argparse.Namespace) -> int:
    from vla_trace.behavior.patchmask import write_patchmask_artifact

    manifest = write_patchmask_artifact(
        args.image,
        args.masks,
        args.output_image,
        variant=args.variant,
        mode=args.mode,
        instances=args.instance or None,
        categories=_parse_key_value_pairs(args.category or []),
        output_manifest=args.output_manifest,
        image_key=args.image_key,
        mask_keys=args.mask_key or None,
        mask_value=args.mask_value,
        bg_ring_width=args.bg_ring_width,
        mosaic_block=args.mosaic_block,
    )
    if args.output_manifest:
        print(str(args.output_manifest))
    else:
        _print_json(manifest)
    return 0


def _cmd_input_edit(args: argparse.Namespace) -> int:
    from vla_trace.behavior.input_edit import (
        build_input_edit_record,
        build_records_from_config,
        summarize_input_edit_results,
        write_input_edit_manifest,
    )

    if args.results:
        summary = summarize_input_edit_results(args.results)
        if args.output:
            write_json(args.output, summary)
            print(str(args.output))
        else:
            _print_json(summary)
        return 0

    if args.config:
        cfg = load_config(args.config)
        _apply_user_path_overrides(cfg, args)
        records = build_records_from_config(cfg)
    else:
        metadata = {
            key: value
            for key, value in {
                "model": args.model,
                "dataset": args.dataset,
                "model_path": args.model_path,
                "data_root": args.data_root,
            }.items()
            if value
        }
        records = [
            build_input_edit_record(
                edit_id=args.edit_id,
                task_id=args.task_id,
                edit_type=args.edit_type,
                base_instruction=args.base_instruction,
                edited_instruction=args.edited_instruction,
                target_object=args.target_object,
                replacement_object=args.replacement_object,
                expected_shift=args.expected_shift,
                metadata=metadata,
            )
        ]
    if not args.output:
        raise ValueError("input-edit manifest generation requires --output")
    output = write_input_edit_manifest(args.output, records)
    print(str(output))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-trace", description="VLA-Trace diagnostic toolkit")
    parser.add_argument("--version", action="version", version=f"vla-trace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_p = sub.add_parser("inspect-model", help="Print model adapter metadata")
    inspect_p.add_argument("config")
    inspect_p.set_defaults(func=_cmd_inspect_model)

    export_libero_p = sub.add_parser("export-libero-manifest", help="Export CKA-ready LIBERO samples and manifest")
    export_libero_p.add_argument("--data-root", required=True, help="User LIBERO RLDS root")
    export_libero_p.add_argument("--data-mix", default="libero_10_no_noops", help="LIBERO RLDS data mix")
    export_libero_p.add_argument("--output-dir", required=True, help="Directory for images/ and manifest.jsonl")
    export_libero_p.add_argument("--max-samples", type=int, default=100)
    export_libero_p.add_argument("--image-size", type=int, default=224)
    export_libero_p.add_argument("--shuffle-buffer-size", type=int, default=500)
    export_libero_p.add_argument("--per-task-quota", type=int)
    export_libero_p.add_argument("--max-stream-reads", type=int)
    export_libero_p.add_argument("--no-train", action="store_true", help="Use evaluation split when supported by the local loader")
    export_libero_p.set_defaults(func=_cmd_export_libero_manifest)

    collect_p = sub.add_parser("collect-repr", help="Collect representation banks from manifest hidden states")
    collect_p.add_argument("config", nargs="?", help="Optional YAML/JSON extraction config")
    collect_p.add_argument("--model", metavar="{OpenVLA,pi0.5}", help=MODEL_HELP)
    collect_p.add_argument("--dataset", metavar="{libero_10,libero_goal,libero_object,libero_spatial}", help=DATASET_HELP)
    collect_p.add_argument("--model-config", help="Path to a user-provided model config YAML/JSON")
    collect_p.add_argument("--benchmark-config", help="Path to a user-provided benchmark config YAML/JSON")
    collect_p.add_argument("--model-path", help="User-provided checkpoint/model directory passed to adapters")
    collect_p.add_argument("--data-root", help="User-provided dataset root passed to adapters")
    collect_p.add_argument("--manifest", help="manifest.jsonl with sample_id/image_path/instruction and hidden_states_path")
    collect_p.add_argument("--hidden-state-dir", help="Directory containing <sample_id>.npz/.npy/.json hidden-state artifacts")
    collect_p.add_argument("--adapter", help="Optional adapter factory import path, e.g. my_pkg.adapters:create_adapter")
    collect_p.add_argument("--bank-output", help="Output representation bank .npz or .json")
    collect_p.add_argument("--checkpoint-name", help="Metadata label such as C0, C1, or C2")
    collect_p.add_argument("--max-samples", type=int)
    collect_p.add_argument("--token-group", action="append", default=[], metavar="NAME=START:STOP", help="Pooling group, e.g. vision_pooled=1:257")
    collect_p.add_argument("--output", help="Optional collection report JSON")
    collect_p.set_defaults(func=_cmd_collect_repr)

    convert_bank_p = sub.add_parser("convert-bank", help="Convert legacy or JSON/NPZ representation banks to public JSON/NPZ")
    convert_bank_p.add_argument("--input", required=True, help="Input bank .pt/.json/.npz")
    convert_bank_p.add_argument("--output", required=True, help="Output bank .json or .npz")
    convert_bank_p.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE", help="Extra metadata to write")
    convert_bank_p.set_defaults(func=_cmd_convert_bank)

    cka_p = sub.add_parser("cka", help="Run Stage 1 CKA analysis on saved banks")
    cka_p.add_argument("config", nargs="?", help="Optional YAML/JSON config")
    cka_p.add_argument("--model", metavar="{OpenVLA,pi0.5}", help=MODEL_HELP)
    cka_p.add_argument("--model-config", help="Path to a user-provided model config YAML/JSON")
    cka_p.add_argument("--model-path", help="User-provided checkpoint/model directory recorded in resolved config")
    cka_p.add_argument("--dataset", metavar="{libero_10,libero_goal,libero_object,libero_spatial}", help=DATASET_HELP)
    cka_p.add_argument("--benchmark-config", help="Path to a user-provided benchmark config YAML/JSON")
    cka_p.add_argument("--data-root", help="User-provided dataset root recorded in resolved config")
    cka_p.add_argument("--analysis", choices=["cross_modal", "checkpoint_drift", "drift"], help="CKA analysis type")
    cka_p.add_argument("--bank", action="append", default=[], metavar="NAME=PATH", help="Saved bank path, e.g. C0=artifacts/c0.json")
    cka_p.add_argument("--layerwise", action="store_true", help="Run layerwise CKA when bank keys include /layer_N")
    cka_p.add_argument("--view", help="Pooled view for checkpoint drift, e.g. joint_pooled")
    cka_p.add_argument("--reference", help="Reference checkpoint name for checkpoint drift")
    cka_p.add_argument("--summary-views", help="Comma-separated matched-layer views for checkpoint drift summaries")
    cka_p.add_argument("--output-dir", help="Directory for the CKA JSON report")
    cka_p.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    cka_p.set_defaults(func=_cmd_cka)

    ko_p = sub.add_parser("knockout", help="Build or summarize Stage 2 knockout sweeps")
    ko_p.add_argument("config", nargs="?", help="Optional YAML/JSON config")
    ko_p.add_argument("--model", metavar="{OpenVLA,pi0.5}", help=MODEL_HELP)
    ko_p.add_argument("--model-config", help="Path to a user-provided model config YAML/JSON")
    ko_p.add_argument("--model-path", help="User-provided checkpoint/model directory recorded in resolved config")
    ko_p.add_argument("--dataset", metavar="{libero_10,libero_goal,libero_object,libero_spatial}", help=DATASET_HELP)
    ko_p.add_argument("--benchmark-config", help="Path to a user-provided benchmark config YAML/JSON")
    ko_p.add_argument("--data-root", help="User-provided dataset root recorded in resolved config")
    ko_p.add_argument("--phase", choices=["prefill", "generation", "both"], help="Knockout phase")
    ko_p.add_argument("--mode", help="Knockout mode, e.g. no_image, no_text, no_vl, no_fusion, or no_image+no_text")
    ko_p.add_argument("--direction", help="Directional knockout, e.g. image->action, text->action, image<->text")
    ko_p.add_argument("--prefill-mode", help="Phase-specific prefill mode for combined settings")
    ko_p.add_argument("--generation-mode", help="Phase-specific generation mode for combined settings")
    ko_p.add_argument(
        "--text-scope",
        choices=["all", "instruction", "semantic_instruction", "full", "bos_newline", "newline_only", "exclude_newline"],
        help="Text token subset used by adapter integrations",
    )
    ko_p.add_argument("--layers", help="all or comma-separated layers, e.g. 0,4,8,12")
    ko_p.add_argument("--center-layers", help="Comma-separated center layers for a window sweep")
    ko_p.add_argument("--window-size", type=int, help="Window size for --center-layers")
    ko_p.add_argument("--num-layers", type=int, help="Total model layers")
    ko_p.add_argument("--visual-tokens", type=int, help="Number of visual tokens")
    ko_p.add_argument("--text-tokens", type=int, help="Number of text tokens")
    ko_p.add_argument("--action-tokens", type=int, help="Number of action tokens")
    ko_p.add_argument("--token-order", help="pi0.5/custom token order, e.g. text,visual,action or visual,text,action")
    ko_p.add_argument("--output")
    ko_p.add_argument("--output-dir", help="Metadata output directory stored in the resolved config")
    ko_p.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    ko_p.set_defaults(func=_cmd_knockout)

    ko_sweep_p = sub.add_parser("knockout-sweep", help="Build standard layerwise knockout job manifests")
    ko_sweep_p.add_argument("--model", required=True, metavar="{OpenVLA,pi0.5}", help=MODEL_HELP)
    ko_sweep_p.add_argument("--dataset", required=True, metavar="{libero_10,libero_goal,libero_object,libero_spatial}", help=DATASET_HELP)
    ko_sweep_p.add_argument("--window-size", type=int, default=5, help="Layer window size, e.g. 3 or 5")
    ko_sweep_p.add_argument("--num-layers", type=int, help="Override model layer count")
    ko_sweep_p.add_argument("--trials", type=int, help="Optional rollout trial count metadata")
    ko_sweep_p.add_argument("--output", required=True, help="Output sweep manifest JSON")
    ko_sweep_p.set_defaults(func=_cmd_knockout_sweep)

    eval_libero_p = sub.add_parser("eval-libero", help="Run or plan LIBERO online rollout evaluation")
    eval_libero_p.add_argument("config", nargs="?", help="Optional VLA-Trace eval YAML/JSON")
    eval_libero_p.add_argument("--model", metavar="{OpenVLA,pi0.5}", help=MODEL_HELP)
    eval_libero_p.add_argument("--dataset", metavar="{libero_10,libero_goal,libero_object,libero_spatial}", help=DATASET_HELP)
    eval_libero_p.add_argument("--model-config", help="Path to a user-provided model metadata config")
    eval_libero_p.add_argument("--benchmark-config", help="Path to a user-provided benchmark metadata config")
    eval_libero_p.add_argument("--model-path", help="User checkpoint/model directory")
    eval_libero_p.add_argument("--data-root", help="User LIBERO data/root directory")
    eval_libero_p.add_argument("--config-path", help="Underlying model/eval config, e.g. a VLM4VLA YAML")
    eval_libero_p.add_argument("--adapter-factory", help="Custom policy factory, e.g. my_pkg.adapters:create_policy")
    eval_libero_p.add_argument("--vlm4vla-root", help="Optional local VLM4VLA checkout for compatibility evaluation")
    eval_libero_p.add_argument("--openpi-root", help="Optional OpenPI root or src directory for pi0.5 adapters")
    eval_libero_p.add_argument("--openpi-config-name", help="OpenPI config name, e.g. pi05_libero")
    eval_libero_p.add_argument("--tokenizer-path", help="Optional tokenizer.model path for pi0.5/OpenPI adapters")
    eval_libero_p.add_argument("--output-dir", help="Directory for result JSON and optional videos")
    eval_libero_p.add_argument("--output", help="Optional extra JSON report path")
    eval_libero_p.add_argument("--result-name", help="Result JSON filename under --output-dir")
    eval_libero_p.add_argument("--task-ids", help="Comma-separated LIBERO task ids, e.g. 0,1,2")
    eval_libero_p.add_argument("--num-trials-per-task", type=int, help="Trials per task")
    eval_libero_p.add_argument("--max-steps", type=int, help="Override LIBERO suite max steps")
    eval_libero_p.add_argument("--num-steps-wait", type=int, help="No-op steps after simulator reset")
    eval_libero_p.add_argument("--execute-step", type=int, help="OpenVLA/VLM4VLA execute_step")
    eval_libero_p.add_argument("--replan-steps", type=int, help="pi0.5 action chunk replan interval")
    eval_libero_p.add_argument("--device", default="cuda", help="Device string passed to adapters")
    eval_libero_p.add_argument("--seed", type=int, default=0)
    eval_libero_p.add_argument("--center-crop", action="store_true", help="Apply OpenVLA center-crop adapter behavior")
    eval_libero_p.add_argument("--save-video", action="store_true", help="Allow saving rollout videos when --video-every is set")
    eval_libero_p.add_argument("--video-every", type=int, help="Save every N episodes; 0 disables videos")
    eval_libero_p.add_argument("--use-openvla-prompt", action="store_true", help="Use official OpenVLA prompt string")
    eval_libero_p.add_argument("--single-unnorm", action="store_true", help="Skip double action unnormalization in compatible adapters")
    eval_libero_p.add_argument("--phase", choices=["prefill", "generation", "both"], help="Knockout phase")
    eval_libero_p.add_argument(
        "--mode",
        default=None,
        help="Knockout mode: baseline, no_image, no_text, no_vl, no_fusion, or a combined mode",
    )
    eval_libero_p.add_argument("--direction", help="Directional knockout, e.g. image->action or text->action")
    eval_libero_p.add_argument("--prefill-mode", help="Phase-specific prefill knockout mode")
    eval_libero_p.add_argument("--generation-mode", help="Phase-specific generation knockout mode")
    eval_libero_p.add_argument(
        "--text-scope",
        choices=["all", "instruction", "semantic_instruction", "full", "bos_newline", "newline_only", "exclude_newline"],
        help="Text token subset for compatible knockout adapters",
    )
    eval_libero_p.add_argument("--layers", help="'all' or comma-separated knockout layers")
    eval_libero_p.add_argument("--center-layer", type=int, help="Layer metadata for plotting one window-scan job")
    eval_libero_p.add_argument("--window-size", type=int, help="Window metadata for plotting one window-scan job")
    eval_libero_p.add_argument("--setting", help="Explicit plot setting label, e.g. generation_no_image")
    eval_libero_p.add_argument("--knockout-manifest", help="Run jobs from a knockout-sweep manifest JSON")
    eval_libero_p.add_argument("--job-index", type=int, help="Run one manifest job by zero-based index")
    eval_libero_p.add_argument("--job-tag", help="Run one manifest job whose tag exactly matches this value")
    eval_libero_p.add_argument("--max-jobs", type=int, help="Limit manifest jobs for smoke tests")
    eval_libero_p.add_argument("--dry-run", action="store_true", help="Write/print the planned jobs without importing model or LIBERO")
    eval_libero_p.add_argument("--mock-env", action="store_true", help="Run a deterministic mock evaluation for CI smoke tests")
    eval_libero_p.add_argument("--print-plan", action="store_true", help="Print resolved plan and exit")
    eval_libero_p.set_defaults(func=_cmd_eval_libero)

    plot_cka_p = sub.add_parser("plot-cka", help="Plot a Stage 1 CKA report JSON")
    plot_cka_p.add_argument("report", help="Path to cross_modal_cka_report.json or checkpoint_drift_cka_report.json")
    plot_cka_p.add_argument("--output", required=True, help="Figure output path, e.g. figures/cka.png")
    plot_cka_p.add_argument("--source-data", help="Optional CSV export of normalized plotted values")
    plot_cka_p.add_argument("--title", help="Optional figure title")
    plot_cka_p.set_defaults(func=_cmd_plot_cka)

    plot_cka_pub_p = sub.add_parser("plot-cka-publication", help="Plot publication-style multi-panel CKA figures")
    plot_cka_pub_p.add_argument("--report", action="append", default=[], metavar="KEY=PATH", help="Report map entry such as openvla:libero_10:alignment=path.json")
    plot_cka_pub_p.add_argument("--output-dir", required=True, help="Directory for publication CKA figures")
    plot_cka_pub_p.add_argument("--datasets", help="Comma-separated datasets, e.g. libero_10,libero_goal")
    plot_cka_pub_p.add_argument("--models", help="Comma-separated models, e.g. pi0.5,OpenVLA")
    plot_cka_pub_p.set_defaults(func=_cmd_plot_cka_publication)

    plot_ko_p = sub.add_parser("plot-knockout", help="Plot LIBERO knockout evaluation result JSON files")
    plot_ko_p.add_argument("inputs", nargs="+", help="Result JSON files or directories containing JSON files")
    plot_ko_p.add_argument("--output", required=True, help="Figure output path, e.g. figures/knockout.png")
    plot_ko_p.add_argument("--model", help="Optional model filter, e.g. OpenVLA or pi0.5")
    plot_ko_p.add_argument("--dataset", help="Optional dataset filter, e.g. libero_10")
    plot_ko_p.add_argument("--setting", help="Optional setting substring filter, e.g. no_image")
    plot_ko_p.add_argument("--source-data", help="Optional CSV export of normalized plotted values")
    plot_ko_p.add_argument("--title", help="Optional figure title")
    plot_ko_p.set_defaults(func=_cmd_plot_knockout)

    plot_ko_line_p = sub.add_parser("plot-knockout-line", help="Plot publication-style layerwise knockout line grids")
    plot_ko_line_p.add_argument("inputs", nargs="*", help="Result JSON files/directories. Omit when using --selected-csv.")
    plot_ko_line_p.add_argument("--selected-csv", help="Source-data CSV main_layerwise_selected.csv")
    plot_ko_line_p.add_argument("--baseline-csv", help="Source-data CSV baselines.csv")
    plot_ko_line_p.add_argument("--output-dir", required=True, help="Directory for line-grid PNG/PDF/SVG outputs")
    plot_ko_line_p.add_argument("--source-data-dir", help="Optional directory for normalized plotted CSV outputs")
    plot_ko_line_p.add_argument("--model", help="Optional model filter, e.g. OpenVLA or pi0.5")
    plot_ko_line_p.add_argument("--dataset", help="Optional dataset filter, e.g. libero_10")
    plot_ko_line_p.add_argument("--protocol", default="window7", help="Publication protocol filter, e.g. window7, window3, non_window")
    plot_ko_line_p.set_defaults(func=_cmd_plot_knockout_line)

    plot_attn_p = sub.add_parser("plot-attention", help="Plot phase-wise attention IoU CSV artifacts")
    plot_attn_p.add_argument("csv", help="CSV with task_id, phase, metric, mean_iou, std_iou, and n_steps")
    plot_attn_p.add_argument("--output", required=True, help="Figure output path, e.g. figures/attention_iou.png")
    plot_attn_p.add_argument("--metric", help="Optional metric filter, e.g. iou_top10")
    plot_attn_p.add_argument("--title", help="Optional figure title")
    plot_attn_p.set_defaults(func=_cmd_plot_attention)

    plot_attn_map_p = sub.add_parser("plot-attention-map", help="Plot generic Stage 3 attention tensors")
    plot_attn_map_p.add_argument("array", help=".npy or .npz attention artifact")
    plot_attn_map_p.add_argument("--output", required=True, help="Figure output path")
    plot_attn_map_p.add_argument("--key", help="Key inside .npz artifacts")
    plot_attn_map_p.add_argument("--keep-last-dims", type=int, choices=[1, 2], default=2)
    plot_attn_map_p.add_argument("--normalize", choices=["none", "sum", "max", "row"], default="none")
    plot_attn_map_p.add_argument("--kind", choices=["auto", "bar", "line", "heatmap"], default="auto")
    plot_attn_map_p.add_argument("--x-labels", help="Comma-separated x-axis labels")
    plot_attn_map_p.add_argument("--y-labels", help="Comma-separated y-axis or series labels")
    plot_attn_map_p.add_argument("--title", help="Optional figure title")
    plot_attn_map_p.set_defaults(func=_cmd_plot_attention_map)

    attn_metrics_p = sub.add_parser("attention-metrics", help="Compute Stage 3 attention localization metrics")
    attn_metrics_p.add_argument("--attention", required=True, help="attention_maps.npz with keys such as step_030")
    attn_metrics_p.add_argument("--masks", required=True, help="step_masks.npz with keys such as step_030_object")
    attn_metrics_p.add_argument("--metadata", help="Optional metadata.json with model, dataset, phases, or split_step")
    attn_metrics_p.add_argument("--objects", help="Optional step_objects.json for category labels")
    attn_metrics_p.add_argument("--output", required=True, help="Per-step metric CSV output")
    attn_metrics_p.add_argument("--summary", help="Optional summary JSON output")
    attn_metrics_p.add_argument("--plot-summary", help="Optional long-form CSV accepted by plot-attention")
    attn_metrics_p.add_argument("--grid-size", type=int, default=16, help="Patch grid side length")
    attn_metrics_p.add_argument("--top-percent", type=float, default=10.0, help="Top-k attention patches for IoU")
    attn_metrics_p.add_argument("--fixed-threshold", type=float, default=0.5, help="Fraction of max attention for fixed-threshold IoU")
    attn_metrics_p.set_defaults(func=_cmd_attention_metrics)

    attn_export_p = sub.add_parser("attention-export", help="Extract qualitative attention views from raw attention tensors")
    attn_export_p.add_argument("--attention", required=True, help="Raw attention .npy/.npz with [layers,heads,query,key]")
    attn_export_p.add_argument("--output", required=True, help="Output .npz containing action_to_image/action_to_text/text_to_image/layer_modality_*")
    attn_export_p.add_argument("--key", help="Key inside input .npz")
    attn_export_p.add_argument("--visual-span", required=True, help="Visual token span START:STOP")
    attn_export_p.add_argument("--text-span", required=True, help="Text token span START:STOP")
    attn_export_p.add_argument("--action-span", required=True, help="Action token span START:STOP")
    attn_export_p.add_argument("--no-normalize-rows", action="store_true", help="Do not row-normalize raw attention before extraction")
    attn_export_p.set_defaults(func=_cmd_attention_export)

    attn_overlay_p = sub.add_parser("attention-overlay", help="Draw image + attention + mask overlay from Stage 3 artifacts")
    attn_overlay_p.add_argument("--image", required=True, help="Image array artifact (.npy or .npz)")
    attn_overlay_p.add_argument("--attention", required=True, help="Attention array artifact (.npy or .npz)")
    attn_overlay_p.add_argument("--masks", help="Optional mask array artifact (.npy or .npz)")
    attn_overlay_p.add_argument("--output", required=True, help="Overlay figure path")
    attn_overlay_p.add_argument("--attention-key", help="Key inside attention .npz")
    attn_overlay_p.add_argument("--mask-key", help="Key inside mask .npz")
    attn_overlay_p.add_argument("--grid-size", type=int, default=16, help="Patch grid side length")
    attn_overlay_p.add_argument("--top-percent", type=float, default=10.0, help="Top-k attention patches to display")
    attn_overlay_p.add_argument("--alpha", type=float, default=0.55, help="Attention overlay alpha")
    attn_overlay_p.set_defaults(func=_cmd_attention_overlay)

    patchmask_p = sub.add_parser("patchmask", help="Generate masked image artifacts for Stage 3 PatchMask probes")
    patchmask_p.add_argument("--image", required=True, help="Input image array artifact (.npy or .npz)")
    patchmask_p.add_argument("--masks", required=True, help="Instance masks artifact (.npy or .npz)")
    patchmask_p.add_argument("--variant", required=True, help="none, mask_target, mask_gripper, mask_robot, mask_robot_exc_gripper, mask_background, or custom")
    patchmask_p.add_argument("--mode", required=True, help="none, black, background_fill, or mosaic")
    patchmask_p.add_argument("--output-image", required=True, help="Masked image output .npy path")
    patchmask_p.add_argument("--output-manifest", help="Optional JSON manifest output")
    patchmask_p.add_argument("--image-key", help="Key inside image .npz")
    patchmask_p.add_argument("--mask-key", action="append", default=[], help="Mask key inside .npz; repeatable")
    patchmask_p.add_argument("--instance", action="append", default=[], help="Instance name for custom/target masks; repeatable")
    patchmask_p.add_argument("--category", action="append", default=[], metavar="INSTANCE=CATEGORY", help="Optional category metadata")
    patchmask_p.add_argument("--mask-value", type=int, default=0)
    patchmask_p.add_argument("--bg-ring-width", type=int, default=8)
    patchmask_p.add_argument("--mosaic-block", type=int, default=8)
    patchmask_p.set_defaults(func=_cmd_patchmask)

    input_edit_p = sub.add_parser("input-edit", help="Build or summarize Stage 3 input-edit manifests/results")
    input_edit_p.add_argument("config", nargs="?", help="Optional YAML/JSON manifest config")
    input_edit_p.add_argument("--output", help="Manifest JSON/JSONL output, or summary JSON with --results")
    input_edit_p.add_argument("--results", help="Summarize input-edit result JSON/JSONL files from a file or directory")
    input_edit_p.add_argument("--model", help=MODEL_HELP)
    input_edit_p.add_argument("--dataset", help=DATASET_HELP)
    input_edit_p.add_argument("--model-path", help="User-provided checkpoint/model directory recorded in manifest metadata")
    input_edit_p.add_argument("--data-root", help="User-provided dataset root recorded in manifest metadata")
    input_edit_p.add_argument("--edit-id", default="edit_0000")
    input_edit_p.add_argument("--task-id", default="")
    input_edit_p.add_argument("--edit-type", default="instruction_replace", help="instruction_replace, object_replace, attribute_replace, or scene_replace")
    input_edit_p.add_argument("--base-instruction", default="")
    input_edit_p.add_argument("--edited-instruction", default="")
    input_edit_p.add_argument("--target-object")
    input_edit_p.add_argument("--replacement-object")
    input_edit_p.add_argument("--expected-shift")
    input_edit_p.set_defaults(func=_cmd_input_edit)

    doctor_p = sub.add_parser("doctor", help="Check local VLA-Trace configs and artifacts")
    doctor_p.add_argument("--model", metavar="{OpenVLA,pi0.5}", help=MODEL_HELP)
    doctor_p.add_argument("--model-config", help="Path to a user-provided model config YAML/JSON")
    doctor_p.add_argument("--dataset", metavar="{libero_10,libero_goal,libero_object,libero_spatial}", help=DATASET_HELP)
    doctor_p.add_argument("--benchmark-config", help="Path to a user-provided benchmark config YAML/JSON")
    doctor_p.add_argument("--model-path", help="User checkpoint/model directory to check")
    doctor_p.add_argument("--data-root", help="User dataset root to check")
    doctor_p.add_argument("--bank", action="append", default=[], metavar="NAME=PATH", help="Representation bank to check")
    doctor_p.add_argument("--attention", help="Attention artifact .npy/.npz to check")
    doctor_p.add_argument("--masks", help="Mask artifact .npy/.npz to check")
    doctor_p.add_argument("--results", help="Knockout/PatchMask result JSON file or directory")
    doctor_p.add_argument("--input-edits", help="Input-edit manifest JSON/JSONL")
    doctor_p.add_argument("--output", help="Optional JSON report output")
    doctor_p.add_argument("--strict", action="store_true", help="Return nonzero when any check has status=error")
    doctor_p.set_defaults(func=_cmd_doctor)

    report_p = sub.add_parser("report", help="Write a lightweight report artifact")
    report_p.add_argument("config")
    report_p.add_argument("--output")
    report_p.set_defaults(func=_cmd_report)
    return parser


def _strict_mask_values(values: np.ndarray, block_value: float = -1.0e9) -> tuple[list, float]:
    """Return strict-JSON mask values by replacing non-finite cells."""
    finite = np.asarray(values, dtype=np.float64).copy()
    finite[np.isneginf(finite)] = block_value
    finite[np.isposinf(finite)] = -block_value
    finite[np.isnan(finite)] = 0.0
    return finite.tolist(), block_value


def _resolve_cka_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config) if args.config else _default_cka_config(args)
    if args.model:
        model = normalize_model(args.model)
        cfg["model"] = model_config_path(model)
    if args.dataset:
        dataset = normalize_dataset(args.dataset)
        cfg["benchmark"] = dataset_config_path(dataset)
    if args.analysis:
        cfg["analysis"] = "checkpoint_drift" if args.analysis == "drift" else args.analysis
    if args.bank:
        cfg["bank_paths"] = _parse_key_value_pairs(args.bank)
    if args.layerwise:
        cfg["layerwise"] = True
    if args.view:
        cfg["view"] = args.view
    if args.reference:
        cfg["reference"] = args.reference
    if args.summary_views:
        cfg["summary_views"] = _parse_str_csv(args.summary_views)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    _apply_user_path_overrides(cfg, args)
    return cfg


def _resolve_knockout_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config) if args.config else _default_knockout_config(args)
    model = normalize_model(args.model) if args.model else _model_from_config(cfg)
    if args.model:
        cfg["model"] = model_config_path(model)
        cfg["family"] = model
        cfg["num_layers"] = DEFAULT_NUM_LAYERS[model]
        cfg["token_layout"] = dict(DEFAULT_TOKEN_LAYOUTS[model])
        cfg["mode"] = DEFAULT_KNOCKOUT_MODE[model]
        cfg["layers"] = {
            "type": "window",
            "center_layers": list(DEFAULT_CENTER_LAYERS[model]),
            "window_size": DEFAULT_WINDOW_SIZE[model],
        }
    if args.dataset:
        dataset = normalize_dataset(args.dataset)
        cfg["benchmark"] = dataset_config_path(dataset)
    if args.phase:
        cfg["phase"] = args.phase
    if args.mode:
        cfg["mode"] = args.mode
    if args.direction:
        cfg["direction"] = args.direction
    if args.prefill_mode:
        cfg["prefill_mode"] = args.prefill_mode
        cfg["phase"] = "both"
    if args.generation_mode:
        cfg["generation_mode"] = args.generation_mode
        cfg["phase"] = "both"
    if args.text_scope:
        cfg["text_scope"] = args.text_scope
    if args.num_layers is not None:
        cfg["num_layers"] = args.num_layers
    if args.layers:
        if args.layers.strip().lower() == "all":
            cfg["layers"] = {"type": "all"}
        else:
            cfg["layers"] = {
                "type": "explicit",
                "values": _parse_int_csv(args.layers),
            }
    elif args.center_layers or args.window_size is not None:
        cfg["layers"] = {
            "type": "window",
            "center_layers": _parse_int_csv(args.center_layers) if args.center_layers else list(DEFAULT_CENTER_LAYERS[model]),
            "window_size": args.window_size if args.window_size is not None else DEFAULT_WINDOW_SIZE[model],
        }
    token_layout = dict(cfg.get("token_layout") or {})
    for arg_name, key in (
        ("visual_tokens", "visual_tokens"),
        ("text_tokens", "text_tokens"),
        ("action_tokens", "action_tokens"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            token_layout[key] = value
    if getattr(args, "token_order", None):
        token_layout["token_order"] = args.token_order
    if token_layout:
        resolved_family = str(cfg.get("family") or model)
        if resolved_family == "pi05" and "token_order" not in token_layout:
            token_layout["token_order"] = "text,visual,action"
        cfg["token_layout"] = token_layout
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    _apply_user_path_overrides(cfg, args)
    return cfg


def _default_cka_config(args: argparse.Namespace) -> dict[str, Any]:
    model = normalize_model(args.model) if args.model else "openvla"
    dataset = normalize_dataset(args.dataset) if args.dataset else "libero_10"
    analysis = args.analysis or "cross_modal"
    if analysis == "drift":
        analysis = "checkpoint_drift"
    return {
        "analysis": analysis,
        "model": model_config_path(model),
        "benchmark": dataset_config_path(dataset),
        "bank_paths": {},
        "output_dir": f"runs/{model}_{dataset}_cka",
        "notes": "Fill --bank/CKA bank_paths with C0/C1/C2 saved representation banks before running a real analysis.",
        **({"summary_views": ["vision_pooled", "text_pooled", "joint_pooled"]} if analysis == "checkpoint_drift" else {}),
    }


def _default_knockout_config(args: argparse.Namespace) -> dict[str, Any]:
    model = normalize_model(args.model) if args.model else "openvla"
    dataset = normalize_dataset(args.dataset) if args.dataset else "libero_10"
    return {
        "model": model_config_path(model),
        "benchmark": dataset_config_path(dataset),
        "family": model,
        "phase": args.phase or "generation",
        "mode": args.mode or DEFAULT_KNOCKOUT_MODE[model],
        "text_scope": args.text_scope or "semantic_instruction",
        "layers": {
            "type": "window",
            "center_layers": list(DEFAULT_CENTER_LAYERS[model]),
            "window_size": DEFAULT_WINDOW_SIZE[model],
        },
        "num_layers": DEFAULT_NUM_LAYERS[model],
        "token_layout": dict(DEFAULT_TOKEN_LAYOUTS[model]),
        "output_dir": f"runs/{model}_{dataset}_knockout",
    }


def _model_from_config(cfg: dict[str, Any]) -> str:
    family = cfg.get("family")
    if family:
        return normalize_model(str(family))
    model_path = str(cfg.get("model", "")).lower()
    for model in MODEL_CONFIGS:
        if model in model_path:
            return model
    return "openvla"


def _parse_key_value_pairs(entries: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=PATH for --bank, got {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected NAME=PATH for --bank, got {entry!r}")
        parsed[key] = value
    return parsed


def _parse_token_group_pairs(entries: list[str]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=START:STOP for --token-group, got {entry!r}")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected NAME=START:STOP for --token-group, got {entry!r}")
        groups[key] = value
    return groups


def _apply_user_path_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    model_config = getattr(args, "model_config", None)
    if model_config:
        cfg["model"] = model_config
    benchmark_config = getattr(args, "benchmark_config", None)
    if benchmark_config:
        cfg["benchmark"] = benchmark_config
    model_path = getattr(args, "model_path", None)
    if model_path:
        cfg["model_path"] = model_path
    data_root = getattr(args, "data_root", None)
    if data_root:
        cfg["data_root"] = data_root


def _parse_int_csv(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Expected at least one comma-separated integer")
    return [int(piece) for piece in pieces]


def _parse_str_csv(value: str) -> list[str]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Expected at least one comma-separated value")
    return pieces


def _resolve_eval_knockout_layers(args: argparse.Namespace, cfg: dict[str, Any]) -> list[int] | str | None:
    value = args.layers if getattr(args, "layers", None) else cfg.get("layers")
    if value is None:
        return None
    if isinstance(value, str):
        return "all" if value.strip().lower() == "all" else _parse_int_csv(value)
    if isinstance(value, dict):
        if value.get("type") == "all":
            return "all"
        if "values" in value:
            return [int(layer) for layer in value["values"]]
        if value.get("type") == "window":
            from vla_trace.knockout.specs import expand_layer_window

            return list(
                expand_layer_window(
                    int(cfg.get("num_layers", 32)),
                    tuple(int(layer) for layer in value.get("center_layers", ())),
                    int(value["window_size"]),
                )
            )
    return [int(layer) for layer in value]


def _parse_task_ids(value: Any) -> tuple[int, ...]:
    if value in (None, "", "all"):
        return ()
    if isinstance(value, str):
        if value.strip().lower() == "all":
            return ()
        return tuple(_parse_int_csv(value))
    return tuple(int(item) for item in value)


def _infer_model_shortcut(value: Any) -> str:
    text = str(value)
    try:
        return normalize_model(text)
    except ValueError:
        lowered = text.lower()
        if "openvla" in lowered:
            return "openvla"
        if "pi05" in lowered or "pi0.5" in lowered or "pi0_5" in lowered:
            return "pi05"
        raise


def _infer_dataset_shortcut(value: Any) -> str:
    text = str(value)
    try:
        return normalize_dataset(text)
    except ValueError:
        lowered = text.lower().replace("-", "_")
        for dataset in DATASET_CONFIGS:
            if dataset in lowered:
                return dataset
        raise


def _select_knockout_manifest_jobs(
    manifest_path: str,
    *,
    job_index: int | None,
    job_tag: str | None,
    max_jobs: int | None,
) -> list[dict[str, Any]]:
    manifest = load_config(manifest_path)
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Expected knockout manifest with a jobs list")
    if job_index is not None:
        if job_index < 0 or job_index >= len(jobs):
            raise ValueError(f"--job-index must be inside [0, {len(jobs)})")
        jobs = [jobs[job_index]]
    if job_tag:
        jobs = [job for job in jobs if str(job.get("tag", "")) == job_tag]
        if not jobs:
            raise ValueError(f"No manifest job matched --job-tag {job_tag!r}")
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    return [dict(job) for job in jobs]


def _knockout_config_from_manifest_job(
    model: str,
    job: dict[str, Any],
    resolver: Any,
) -> dict[str, Any] | None:
    knockout_config = resolver(
        model=model,
        phase=str(job.get("phase", "generation")),
        mode=str(job.get("mode", "baseline")),
        layers=job.get("layers"),
        direction=job.get("direction"),
        text_scope=str(job.get("text_scope", "all")),
    )
    if isinstance(knockout_config, dict):
        if job.get("center_layer") is not None:
            knockout_config["center_layer"] = int(job["center_layer"])
        if job.get("window_size") is not None:
            knockout_config["window_size"] = int(job["window_size"])
    return knockout_config


def _print_json(payload: Any) -> None:
    print(json.dumps(_strict_json_payload(payload), indent=2, ensure_ascii=False, allow_nan=False))


def _display_model(cfg: dict[str, Any]) -> str:
    return _model_from_config(cfg)


def _display_dataset(cfg: dict[str, Any]) -> str:
    benchmark = str(cfg.get("benchmark", "")).lower()
    for dataset in DATASET_CONFIGS:
        if dataset in benchmark:
            return dataset
    return "unknown"


def _strict_json_payload(value: Any) -> Any:
    """Convert non-finite metadata values to JSON-standard string markers."""
    if isinstance(value, dict):
        return {key: _strict_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_payload(item) for item in value]
    if isinstance(value, float):
        if np.isneginf(value):
            return "-inf"
        if np.isposinf(value):
            return "inf"
        if np.isnan(value):
            return "nan"
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
