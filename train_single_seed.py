from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


SCENARIOS = ("mixed_10", "mixed_20", "mixed_30")
HORIZONS = (8, 16, 24, 48)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure(package_root, horizon, scenario, *, seed=2026, device="cuda", smoke=False, backbone=None):
    root = Path(package_root).resolve()
    if horizon not in HORIZONS or scenario not in SCENARIOS:
        raise ValueError("Unsupported Table 3 condition")
    run = root / "runs" / f"h{horizon}" / scenario
    config = read_json(run / "model_config.json")
    protocol = read_json(run / "training_config.json")
    required = ("warmup", "joint", "quality", "graph")
    if any(name not in protocol for name in required):
        raise ValueError(
            "training_config.json must describe the completed fine-tuning recipe. "
            "For Mixed-3/H48 use the three-seed series warmup/calibration recipe, "
            "not the archived zero-extra-fine-tuning control."
        )
    recipe = protocol["joint"]["candidate"]
    expected = ("original_single_scene_protocol" if horizon in (8, 16)
                else "control_warmup_b050" if scenario == "mixed_30"
                else "low_lr" if horizon == 24 else "restore_05")
    if recipe != expected:
        raise ValueError(f"Expected frozen recipe {expected!r}, found {recipe!r}")
    if config.get("pred_len") != horizon or config.get("scenario_name") != scenario:
        raise ValueError("Packaged model configuration and requested condition differ")
    data = root / "data" / "pv_power_selected_aux_7features.csv"
    backbone_override = backbone is not None
    backbone = Path(backbone).resolve() if backbone_override else run / "backbone.pth"
    for path in (root / "main_mprgnetv2_single_scene.py", data, backbone):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_sha = sha256(backbone)
    expected_sha = config.get("actual_backbone_sha256")
    if not backbone_override and expected_sha and actual_sha != expected_sha:
        raise ValueError("Packaged backbone does not match its archived SHA-256")
    args = SimpleNamespace(**config)
    args.seed = int(seed)
    args.data_seed = int(protocol.get("data_seed", 2026))
    if args.data_seed != 2026:
        raise ValueError("This release uses data_seed=2026")
    args.pred_len, args.scenario_name = horizon, scenario
    args.seeds, args.scenarios = [args.seed], [scenario]
    args.device, args.data = device, str(data)
    args.initialization = "single_scene_backbone"
    args.initial_checkpoint_dir = args.evaluation_checkpoint_dir = None
    args.resume_run, args.smoke = None, bool(smoke)
    args.backbone_checkpoint_template = str(backbone)
    args.mixed30_backbone_checkpoint_template = str(backbone)
    args.actual_backbone_checkpoint = str(backbone)
    args.actual_backbone_sha256 = actual_sha
    args.batch_size = int(protocol["batch_size"])
    args.grad_clip = float(protocol["grad_clip"])
    args.weight_decay = float(protocol["optimizer"]["weight_decay"])
    args.num_threads = int(protocol.get("num_threads", 4))
    args.num_workers = int(protocol.get("num_workers", 0))
    warmup, joint, quality, graph = (protocol[key] for key in required)
    args.warmup_epochs, args.warmup_lr = int(warmup["epochs"]), float(warmup["lr"])
    args.warmup_windows = int(warmup["windows"])
    args.graph_guard_weight = float(warmup.get("guard_weight", 0.5))
    args.warmup_history_weight = float(warmup.get("history_weight", 0.05))
    args.epochs = int(joint.get("max_epochs", 8))
    args.lr = float(joint.get("lr", getattr(args, "lr", 1e-4)))
    args.joint_graph_lr = float(joint.get("graph_lr", getattr(args, "joint_graph_lr", 3e-4)))
    args.patience = int(joint.get("early_stopping", {}).get("patience", 4))
    args.decay_patience = int(joint.get("scheduler", {}).get("patience", 2))
    args.train_windows, args.validation_windows = int(joint.get("train_windows", 0)), int(joint.get("validation_windows", 0))
    args.quality_epochs, args.quality_lr = int(quality["epochs"]), float(quality["lr"])
    args.quality_windows = int(quality["train_windows"])
    args.quality_validation_windows = int(quality["validation_windows"])
    args.graph_epochs, args.graph_lr = int(graph["epochs"]), float(graph["lr"])
    args.graph_windows = int(graph["windows"])
    args.select_message_bound = float(graph["message_bound"])
    args.jointq_mode, args.jointq_scope, args.jointq_strength = "soft", "longgap", 0.5
    args.jointq_gap_steps = int(graph.get("quality_gap_steps", 24))
    if smoke:
        args.warmup_epochs = args.epochs = args.quality_epochs = args.graph_epochs = 1
        args.train_windows = args.warmup_windows = args.quality_windows = args.graph_windows = 32
        args.validation_windows, args.quality_validation_windows = 96, 32
    return args, protocol, recipe, run


def run_recipe(package_root, horizon, scenario, output, *, seed=2026, device="cuda", smoke=False, evaluate=True, backbone=None):
    root = Path(package_root).resolve()
    sys.path.insert(0, str(root))
    from reproduce import ensure_dataset
    ensure_dataset(root)
    args, protocol, recipe, archive = configure(root, horizon, scenario, seed=seed, device=device, smoke=smoke, backbone=backbone)
    output = Path(output).resolve()
    if output == root or root in output.parents and output.parts[len(root.parts)] in ("model", "Dataprovider", "data", "runs", "layers", "utils"):
        raise ValueError("Choose an output directory outside the packaged input directories")
    output.mkdir(parents=True, exist_ok=False)
    args.output_root = str(output)
    import torch
    import main_mprgnetv2_single_scene as single
    import main_mprgnetv2_table3 as table3
    import main_mprgnetv2_tune_long as tuning
    import main_mprgnetv2_tune_mixed30 as round2
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu to use CPU")
    torch.set_num_threads(args.num_threads)
    started = time.perf_counter()
    manifest = output / "summary.json"
    record = dict(status="preparing", seed=args.seed, data_seed=args.data_seed,
                  backbone_seed=2026 if backbone is None else None, horizon=horizon, scenario=scenario,
                  recipe=recipe, smoke_only=bool(smoke), configuration=vars(args).copy(),
                  frozen_training_protocol=protocol, backbone_sha256=args.actual_backbone_sha256,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  selection="Frozen candidate and training budgets; validation-only checkpoint selection; test after freezing.")
    single.save_json(manifest, record)
    original_assess, original_guards = tuning.assess, copy.deepcopy(tuning.GUARDS)
    try:
        single.seed_all(args.seed)
        training, validation, test, meta = single.build_data(args)
        record.update(status="training", data=meta["effective_data_config"])
        single.save_json(manifest, record)
        if recipe == "original_single_scene_protocol":
            final = single.train_one(training, validation, meta, args, output)
            checkpoint = output / "best.pt"
        else:
            tuning.GUARDS.clear()
            tuning.GUARDS.update(forecast_ratio=1.02, reconstruction_ratio=1.10)
            tuning.assess = round2.ORIGINAL_ASSESS
            if recipe == "control_warmup_b050":
                round2.configure_selection()
            for key, value in tuning.CANDIDATES.get(recipe, {}).items():
                if protocol["joint"].get(key) != value:
                    raise ValueError(f"Frozen recipe mismatch for {recipe}/{key}")
            model = single.initialize_backbone(args)
            model.set_relation_prior(single.relation_prior(training), split_name="train")
            scale = torch.as_tensor(meta["pv_scaler"].std, device=args.device,
                                    dtype=torch.float32).reshape(1, 1, -1)
            stages = dict(seed=args.seed, source_checkpoint=args.actual_backbone_checkpoint, warmup=[])
            single.graph_warmup(model, training, validation, scale, args, output, stages)
            del model
            reference = tuning.prepare_reference(output, output / "warmup.pt", args, validation, scale)
            if recipe in tuning.CANDIDATES:
                trial_cli = SimpleNamespace(smoke=bool(smoke), epochs=args.epochs)
                selected = tuning.train_candidate(trial_cli, recipe, output / recipe,
                    output / "warmup.pt", output / "reference.pt", reference,
                    training, validation, args, scale)
            else:
                selected = round2.cap_control(output / recipe, recipe, output / "warmup.pt",
                    0.5, args, validation, scale, reference)
                if not selected["assessment"]["eligible"]:
                    selected = dict(candidate="reference_fallback_b050", checkpoint=str(output / "reference.pt"))
            record.update(status="calibrating_and_adapting", selected_start=selected["candidate"])
            single.save_json(manifest, record)
            final = tuning.calibrate_and_adapt(SimpleNamespace(smoke=bool(smoke)), output,
                selected, reference, training, validation, args, scale)
            checkpoint = Path(final["checkpoint"])
        if not final.get("complete", final.get("training_complete", False)):
            raise RuntimeError("Training did not complete")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["seed"] != args.seed or saved["source_sha256"] != args.actual_backbone_sha256:
            raise RuntimeError("Trained checkpoint provenance mismatch")
        frozen_sha = single.sha(checkpoint)
        record.update(status="selection_frozen", training=final,
                      checkpoint=str(checkpoint), frozen_checkpoint_sha256=frozen_sha)
        single.save_json(manifest, record)
        if evaluate:
            timing = SimpleNamespace(smoke=bool(smoke), timing_batch_size=64, timing_repetitions=1 if smoke else 3)
            record["evaluation"] = table3.evaluate_checkpoint(checkpoint, test, meta, args, timing)
            if single.sha(checkpoint) != frozen_sha:
                raise RuntimeError("Frozen checkpoint changed during evaluation")
        record.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                      completed_at=datetime.now(timezone.utc).isoformat())
        single.save_json(manifest, record)
        return record
    except BaseException as error:
        record.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        single.save_json(manifest, record)
        raise
    finally:
        tuning.assess = original_assess
        tuning.GUARDS.clear()
        tuning.GUARDS.update(original_guards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--smoke", action="store_true", help="One epoch per stage; tiny train/validation subsets")
    parser.add_argument("--no-test", action="store_true", help="Train and freeze without test evaluation")
    cli = parser.parse_args()
    result = run_recipe(cli.package_root, cli.horizon, cli.scenario, cli.output,
                        seed=cli.seed, device=cli.device, smoke=cli.smoke, evaluate=not cli.no_test, backbone=cli.backbone)
    print("CHECKPOINT=" + result["checkpoint"], flush=True)
    if "evaluation" in result:
        print("METRICS=" + json.dumps(result["evaluation"]["metrics"]), flush=True)


if __name__ == "__main__":
    main()
