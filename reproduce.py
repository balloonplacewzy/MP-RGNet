import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_dataset(package_root=None):
    root = Path(package_root).resolve() if package_root is not None else ROOT
    info = read(root / "manifest.json")["dataset"]
    archive, target = root / info["archive"], root / info["path"]
    if target.exists():
        if sha(target) != info["sha256"]:
            raise ValueError("Dataset checksum mismatch: " + str(target))
        return target
    if sha(archive) != info["archive_sha256"]:
        raise ValueError("Dataset archive checksum mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.tmp")
    with zipfile.ZipFile(archive) as bundle, bundle.open(info["member"]) as source, temporary.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    if sha(temporary) != info["sha256"]:
        temporary.unlink()
        raise ValueError("Extracted dataset checksum mismatch")
    temporary.replace(target)
    return target


def evaluate_one(key, device, checkpoint=None, max_windows=0):
    from types import SimpleNamespace
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "source"))
    import main_mprgnetv2_single_scene as single
    from main_ours import compute_metrics, inverse_pv_scale
    from models.MPRGNetv2 import Model
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    torch.set_num_threads(4)
    single.seed_all(2026)
    row = read(ROOT / "manifest.json")["runs"][key]
    path = Path(checkpoint).resolve() if checkpoint else ROOT / row["checkpoint"]
    if not checkpoint and sha(path) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch: " + key)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    args = SimpleNamespace(**saved["config"])
    args.data, args.device, args.num_workers = str(ensure_dataset()), device, 0
    if args.pred_len != row["horizon"] or args.scenario_name != row["scenario"]:
        raise ValueError("Checkpoint and requested condition differ")
    training, validation, test, meta = single.build_data(args)
    del training, validation
    model = Model(args).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.mode = "soft"
    model.set_stage("eval").eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for raw in single.loader(test, args, max_windows):
            batch = single.move(raw, device)
            prediction = model(batch["x"])["pred"]
            if prediction.shape != batch["y"].shape or not torch.isfinite(prediction).all():
                raise ValueError("Invalid model predictions")
            predictions.append(prediction.cpu().numpy())
            targets.append(batch["y"].cpu().numpy())
    pred = inverse_pv_scale(np.concatenate(predictions), meta["pv_scaler"], args.target_dim)
    truth = inverse_pv_scale(np.concatenate(targets), meta["pv_scaler"], args.target_dim)
    result = dict(run=key, device=device, checkpoint_sha256=sha(path),
                  metrics=compute_metrics(pred, truth), test_windows=len(pred),
                  full_test=(len(pred) == len(test)))
    del model, pred, truth, predictions, targets, test, meta
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    train = commands.add_parser("train")
    for child in (evaluate, train):
        group = child.add_mutually_exclusive_group(required=True)
        group.add_argument("--run")
        group.add_argument("--all", action="store_true")
        child.add_argument("--device", default="cuda")
        child.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--max-windows", type=int, default=0)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--backbone", type=Path)
    args = parser.parse_args()
    manifest = read(ROOT / "manifest.json")
    keys = list(manifest["runs"]) if args.all else [args.run]
    if any(key not in manifest["runs"] for key in keys):
        parser.error("Run must be one of: " + ", ".join(manifest["runs"]))
    if args.command == "evaluate" and args.checkpoint and args.all:
        parser.error("--checkpoint requires a single --run")
    if args.command == "evaluate" and args.max_windows < 0:
        parser.error("--max-windows cannot be negative")
    if args.command == "train" and args.backbone and args.all:
        parser.error("--backbone requires a single --run")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output path")
    if ROOT in output.parents and output.parts[len(ROOT.parts)] in ("source", "data", "runs"):
        parser.error("Output must be outside source/, data/ and runs/")
    if args.command == "evaluate":
        result = dict(status="running", scope="complete test" if not args.max_windows else "smoke subset", runs={})
        for key in keys:
            result["runs"][key] = evaluate_one(key, args.device, args.checkpoint, args.max_windows)
            write(output, result)
            print(key + " " + json.dumps(result["runs"][key]["metrics"]), flush=True)
        result["status"] = "completed"
        write(output, result)
    else:
        ensure_dataset()
        for key in keys:
            horizon, scenario = key.split("/")
            command = [sys.executable, str(ROOT / "train_single_seed.py"), "--horizon", horizon[1:],
                       "--scenario", scenario, "--device", args.device, "--output", str(output / key)]
            if args.smoke:
                command.append("--smoke")
            if args.backbone:
                command.extend(["--backbone", str(args.backbone.resolve())])
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = "2026"
            subprocess.run(command, cwd=str(ROOT), env=environment, check=True)


if __name__ == "__main__":
    main()
