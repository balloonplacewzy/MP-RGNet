import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
RUNS = tuple(f"h{h}/{scenario}" for h in (8, 16, 24, 48)
             for scenario in ("mixed_10", "mixed_20", "mixed_30"))
DATA_FILE = "pv_power_selected_aux_7features.csv"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ensure_dataset(package_root=None):
    root = Path(package_root).resolve() if package_root is not None else ROOT
    target = root / "data" / DATA_FILE
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".csv.tmp")
    with zipfile.ZipFile(root / "dataset.zip") as bundle:
        with bundle.open("dataset/" + DATA_FILE) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    temporary.replace(target)
    return target


def evaluate_one(key, device, checkpoint=None, max_windows=0):
    from types import SimpleNamespace
    import numpy as np
    import torch
    import main_mprgnetv2_single_scene as single
    from main_ours import compute_metrics, inverse_pv_scale
    from model.MPRGNetv2 import Model
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    torch.set_num_threads(4)
    single.seed_all(2026)
    horizon, scenario = key.split("/")
    path = Path(checkpoint).resolve() if checkpoint else ROOT / "runs" / key / "model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    args = SimpleNamespace(**saved["config"])
    args.data, args.device, args.num_workers = str(ensure_dataset()), device, 0
    if args.pred_len != int(horizon[1:]) or args.scenario_name != scenario:
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
    result = dict(run=key, device=device, checkpoint=str(path),
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
        group.add_argument("--run", choices=RUNS)
        group.add_argument("--all", action="store_true")
        child.add_argument("--device", default="cuda")
        child.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--max-windows", type=int, default=0)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--backbone", type=Path)
    args = parser.parse_args()
    keys = RUNS if args.all else (args.run,)
    if args.command == "evaluate" and args.checkpoint and args.all:
        parser.error("--checkpoint requires a single --run")
    if args.command == "evaluate" and args.max_windows < 0:
        parser.error("--max-windows cannot be negative")
    if args.command == "train" and args.backbone and args.all:
        parser.error("--backbone requires a single --run")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output path")
    if output == ROOT or (ROOT in output.parents and output.parts[len(ROOT.parts)] in
                          ("model", "Dataprovider", "data", "runs", "layers", "utils")):
        parser.error("Choose an output path outside the code, data and runs directories")
    if args.command == "evaluate":
        result = dict(status="running", runs={})
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
