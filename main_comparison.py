import argparse
import importlib
import inspect
import os
import random
import time
from pathlib import Path
from numbers import Number
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from get_config import get_config
from data_provider.pv_quality_dataset import PVDataConfig, build_pv_quality_dataloaders


def fix_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global Random Seed Fixed: {seed}")


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_model(args):

    try:
        module = importlib.import_module(f"models.{args.model_name}")
    except ModuleNotFoundError:
        module = importlib.import_module(f"models.{args.model_name.lower()}")
    return module.Model(args)


def model_accepts_batch(model: nn.Module) -> bool:


    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False

    if "batch" in sig.parameters:
        return True

    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_criterion(loss_name: str):
    name = loss_name.lower()
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unknown loss: {loss_name}")


def move_batch_to_device(batch, device):


    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=True)
            if value.dtype.is_floating_point:
                value = value.float()
            moved[key] = value
        else:
            moved[key] = value
    return moved


def call_model(model, batch, args, pass_batch: bool):


    x = batch["x"]
    if args.model_type == "ours" and pass_batch:
        return model(x, batch=batch)
    return model(x)


def parse_model_output(model_output):


    if isinstance(model_output, torch.Tensor):
        return model_output, {}

    if isinstance(model_output, (tuple, list)):
        if len(model_output) == 0:
            raise ValueError("Model returned an empty tuple/list.")
        pred = model_output[0]
        loss_dict = model_output[1] if len(model_output) > 1 and isinstance(model_output[1], dict) else {}
        if not isinstance(pred, torch.Tensor):
            raise TypeError("The first item returned by the model must be a prediction Tensor.")
        return pred, loss_dict

    if isinstance(model_output, dict):
        pred = None
        pred_keys = ("pred", "outputs", "output", "y_hat", "forecast")
        used_pred_key = None
        for key in pred_keys:
            if key in model_output:
                pred = model_output[key]
                used_pred_key = key
                break
        if pred is None:
            raise KeyError(
                "Model output dict must contain one of: " + ", ".join(pred_keys)
            )
        if not isinstance(pred, torch.Tensor):
            raise TypeError("Prediction entry in model output dict must be a Tensor.")

        loss_dict = {}
        if isinstance(model_output.get("losses"), dict):
            loss_dict.update(model_output["losses"])


        ignored = set(pred_keys) | {"losses"}
        if used_pred_key is not None:
            ignored.add(used_pred_key)
        for key, value in model_output.items():
            if key in ignored:
                continue
            if torch.is_tensor(value) or isinstance(value, Number):
                loss_dict[key] = value

        return pred, loss_dict

    raise TypeError(
        "Unsupported model output type. Expected Tensor, tuple/list, or dict; "
        f"got {type(model_output)}."
    )


def _to_loss_tensor(value, reference: torch.Tensor):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    if isinstance(value, Number):
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    raise TypeError(f"Auxiliary loss must be a Tensor or number, got {type(value)}")


def compute_total_loss(model_output, batch_y, criterion, args):


    pred, loss_dict = parse_model_output(model_output)
    pred_loss = criterion(pred, batch_y)
    total_loss = pred_loss


    loss_specs = {
        "impute_loss": "lambda_impute",
        "unc_loss": "lambda_unc",
        "uncertainty_loss": "lambda_unc",
        "smooth_loss": "lambda_smooth",
        "smoothness_loss": "lambda_smooth",
        "graph_loss": "lambda_graph",
    }

    used_aux = {}
    for loss_key, weight_name in loss_specs.items():
        if loss_key not in loss_dict:
            continue
        weight = float(getattr(args, weight_name, 0.0))
        if weight == 0.0:
            continue
        aux_loss = _to_loss_tensor(loss_dict[loss_key], pred_loss)
        total_loss = total_loss + weight * aux_loss
        used_aux[loss_key] = aux_loss.detach()

    return total_loss, pred, pred_loss.detach(), used_aux


def compute_metrics(preds, trues, day_threshold=None):


    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)

    err = preds - trues
    mae = np.mean(np.abs(err))
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum(np.abs(trues)) + 1e-8
    wape = np.sum(np.abs(err)) / denom * 100.0

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((trues - np.mean(trues)) ** 2) + 1e-8
    r2 = 1.0 - ss_res / ss_tot

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "R2": r2,
        "Count": trues.size,
    }

    if day_threshold is not None:
        mask = trues > day_threshold
        if np.any(mask):
            day_err = err[mask]
            day_true = trues[mask]
            day_mae = np.mean(np.abs(day_err))
            day_rmse = np.sqrt(np.mean(day_err ** 2))
            day_wape = np.sum(np.abs(day_err)) / (np.sum(np.abs(day_true)) + 1e-8) * 100.0
            day_ss_res = np.sum(day_err ** 2)
            day_ss_tot = np.sum((day_true - np.mean(day_true)) ** 2) + 1e-8
            day_r2 = 1.0 - day_ss_res / day_ss_tot
            metrics.update({
                "Day_MAE": day_mae,
                "Day_RMSE": day_rmse,
                "Day_WAPE": day_wape,
                "Day_R2": day_r2,
                "Day_Count": int(mask.sum()),
            })
        else:
            metrics.update({
                "Day_MAE": np.nan,
                "Day_RMSE": np.nan,
                "Day_WAPE": np.nan,
                "Day_R2": np.nan,
                "Day_Count": 0,
            })

    return metrics


def inverse_pv_scale(data: np.ndarray, pv_scaler, target_dim: int) -> np.ndarray:


    original_shape = data.shape
    data_2d = data.reshape(-1, target_dim)
    inv_2d = pv_scaler.inverse_transform(data_2d)
    return inv_2d.reshape(original_shape)


def _empty_imputation_metrics(prefix: str, source: str, reason: str = ""):
    return {
        f"{prefix}_MAE": np.nan,
        f"{prefix}_RMSE": np.nan,
        f"{prefix}_Count": 0,
        f"{prefix}_Source": source,
        f"{prefix}_Space": "raw",
        f"{prefix}_Reason": reason,
    }


def compute_raw_imputation_metrics(
    imputed_scaled: np.ndarray,
    clean_scaled: np.ndarray,
    obs_mask: np.ndarray,
    pv_scaler,
    target_dim: int,
    eval_mask: np.ndarray = None,
):


    imputed_scaled = np.asarray(imputed_scaled, dtype=np.float32)
    clean_scaled = np.asarray(clean_scaled, dtype=np.float32)
    obs_mask = np.asarray(obs_mask, dtype=np.float32)

    if imputed_scaled.shape != clean_scaled.shape or imputed_scaled.shape != obs_mask.shape:
        raise ValueError(
            "Imputation metric arrays must have the same shape: "
            f"imputed={imputed_scaled.shape}, clean={clean_scaled.shape}, mask={obs_mask.shape}"
        )

    missing_mask = obs_mask < 0.5
    if eval_mask is not None:
        eval_mask = np.asarray(eval_mask, dtype=bool)
        if eval_mask.shape != missing_mask.shape:
            raise ValueError(
                f"eval_mask shape {eval_mask.shape} does not match data shape {missing_mask.shape}"
            )
        missing_mask = missing_mask & eval_mask

    count = int(missing_mask.sum())
    if count == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "Count": 0}

    imputed_raw = inverse_pv_scale(imputed_scaled, pv_scaler, target_dim=target_dim)
    clean_raw = inverse_pv_scale(clean_scaled, pv_scaler, target_dim=target_dim)

    err = imputed_raw[missing_mask] - clean_raw[missing_mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    return {"MAE": mae, "RMSE": rmse, "Count": count}


def evaluate_input_imputation_metrics(test_loader, pv_scaler, target_dim: int, source: str):


    dataset = test_loader.dataset
    split_start = int(dataset.split_start)

    hist_end = int(dataset.split_end) - int(dataset.pred_len)
    if hist_end <= split_start:
        metrics = _empty_imputation_metrics("Input_Impute", source, reason="empty_test_history")
    else:
        imputed = dataset.pv_obs_input[split_start:hist_end]
        clean = dataset.y_clean[split_start:hist_end]
        obs_mask = dataset.obs_mask_full[split_start:hist_end]
        raw = compute_raw_imputation_metrics(
            imputed_scaled=imputed,
            clean_scaled=clean,
            obs_mask=obs_mask,
            pv_scaler=pv_scaler,
            target_dim=target_dim,
        )
        metrics = {
            "Input_Impute_MAE": raw["MAE"],
            "Input_Impute_RMSE": raw["RMSE"],
            "Input_Impute_Count": raw["Count"],
            "Input_Impute_Source": source,
            "Input_Impute_Space": "raw",
            "Input_Impute_Reason": "obs_mask<0.5",
        }


    if source == "knn":
        metrics.update({
            "KNN_Impute_MAE": metrics["Input_Impute_MAE"],
            "KNN_Impute_RMSE": metrics["Input_Impute_RMSE"],
            "KNN_Impute_Count": metrics["Input_Impute_Count"],
            "KNN_Impute_Space": "raw",
        })

    metrics.update({
        "Impute_MAE": metrics["Input_Impute_MAE"],
        "Impute_RMSE": metrics["Input_Impute_RMSE"],
        "Impute_Count": metrics["Input_Impute_Count"],
        "Impute_Source": source,
        "Impute_Space": "raw",
    })
    return metrics


IMPUTATION_OUTPUT_KEYS = (
    "impute",
    "imputed",
    "imputation",
    "p_impute",
    "P_impute",
    "pv_impute",
    "pv_imputed",
    "p_fuse",
    "P_fuse",
    "pv_fuse",
    "p_corr",
    "P_corr",
    "pv_corr",
    "reconstruction",
    "recon",
    "pv_recon",
    "x_hat",
    "pv_hat",
)


def extract_imputation_tensor(model_output):


    if isinstance(model_output, dict):
        for key in IMPUTATION_OUTPUT_KEYS:
            value = model_output.get(key, None)
            if torch.is_tensor(value):
                return value, key

        for nested_key in ("extra", "extras", "aux", "auxiliary"):
            nested = model_output.get(nested_key, None)
            if isinstance(nested, dict):
                for key in IMPUTATION_OUTPUT_KEYS:
                    value = nested.get(key, None)
                    if torch.is_tensor(value):
                        return value, f"{nested_key}.{key}"

    if isinstance(model_output, (tuple, list)):

        for item in model_output[1:]:
            if isinstance(item, dict):
                for key in IMPUTATION_OUTPUT_KEYS:
                    value = item.get(key, None)
                    if torch.is_tensor(value):
                        return value, key

        if len(model_output) >= 3 and torch.is_tensor(model_output[2]):
            return model_output[2], "tuple[2]"

    return None, None


def normalize_imputation_tensor(impute_tensor: torch.Tensor, batch, target_dim: int):

    if not torch.is_tensor(impute_tensor):
        raise TypeError("impute_tensor must be a torch.Tensor.")
    if impute_tensor.dim() != 3:
        raise ValueError(f"Expected imputation tensor with 3 dims [B, L, N], got {tuple(impute_tensor.shape)}")

    seq_len = int(batch["pv_clean"].shape[1])

    if impute_tensor.shape[1] == seq_len and impute_tensor.shape[2] == target_dim:
        return impute_tensor


    if impute_tensor.shape[1] == target_dim and impute_tensor.shape[2] == seq_len:
        return impute_tensor.transpose(1, 2)

    raise ValueError(
        "Imputation tensor shape is not compatible with historical PV shape. "
        f"Got {tuple(impute_tensor.shape)}, expected [B, {seq_len}, {target_dim}] "
        f"or [B, {target_dim}, {seq_len}]."
    )


@torch.no_grad()
def evaluate_model_imputation_metrics(model, test_loader, pv_scaler, device, target_dim: int, args, pass_batch):


    model.eval()
    dataset = test_loader.dataset
    total_len = int(dataset.y_clean.shape[0])

    sum_impute = np.zeros((total_len, target_dim), dtype=np.float64)
    count_impute = np.zeros((total_len, target_dim), dtype=np.float64)
    used_key = None

    for raw_batch in test_loader:
        batch = move_batch_to_device(raw_batch, device)
        model_output = call_model(model, batch, args, pass_batch)
        impute_tensor, key = extract_imputation_tensor(model_output)
        if impute_tensor is None:
            metrics = _empty_imputation_metrics(
                "Model_Impute",
                source="model_output",
                reason="no_imputation_tensor_returned",
            )
            metrics.update({
                "Impute_MAE": np.nan,
                "Impute_RMSE": np.nan,
                "Impute_Count": 0,
                "Impute_Source": "model_output",
                "Impute_Space": "raw",
            })
            return metrics

        used_key = key
        impute_tensor = normalize_imputation_tensor(impute_tensor, batch, target_dim=target_dim)
        impute_np = impute_tensor.detach().cpu().numpy().astype(np.float32)
        starts = batch["index"].detach().cpu().numpy().astype(np.int64)

        for b, s in enumerate(starts):
            e = int(s) + impute_np.shape[1]
            sum_impute[int(s):e] += impute_np[b]
            count_impute[int(s):e] += 1.0

    covered = count_impute > 0
    if not np.any(covered):
        metrics = _empty_imputation_metrics("Model_Impute", source="model_output", reason="no_covered_history")
    else:
        avg_impute = np.zeros_like(sum_impute, dtype=np.float32)
        avg_impute[covered] = (sum_impute[covered] / count_impute[covered]).astype(np.float32)

        raw = compute_raw_imputation_metrics(
            imputed_scaled=avg_impute,
            clean_scaled=dataset.y_clean,
            obs_mask=dataset.obs_mask_full,
            pv_scaler=pv_scaler,
            target_dim=target_dim,
            eval_mask=covered,
        )
        metrics = {
            "Model_Impute_MAE": raw["MAE"],
            "Model_Impute_RMSE": raw["RMSE"],
            "Model_Impute_Count": raw["Count"],
            "Model_Impute_Source": f"model_output:{used_key}",
            "Model_Impute_Space": "raw",
            "Model_Impute_Reason": "obs_mask<0.5",
        }

    metrics.update({
        "Impute_MAE": metrics["Model_Impute_MAE"],
        "Impute_RMSE": metrics["Model_Impute_RMSE"],
        "Impute_Count": metrics["Model_Impute_Count"],
        "Impute_Source": metrics["Model_Impute_Source"],
        "Impute_Space": "raw",
    })
    return metrics


def evaluate_imputation_metrics(model, test_loader, pv_scaler, device, target_dim: int, args, pass_batch):

    if args.model_type == "ours":
        return evaluate_model_imputation_metrics(
            model=model,
            test_loader=test_loader,
            pv_scaler=pv_scaler,
            device=device,
            target_dim=target_dim,
            args=args,
            pass_batch=pass_batch,
        )

    return evaluate_input_imputation_metrics(
        test_loader=test_loader,
        pv_scaler=pv_scaler,
        target_dim=target_dim,
        source=str(args.impute_method),
    )


def train_one_epoch(model, train_loader, criterion, optimizer, device, args, pass_batch, scaler=None):
    model.train()
    losses = []
    pred_losses = []
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"

    for raw_batch in train_loader:
        batch = move_batch_to_device(raw_batch, device)
        batch_y = batch["y"]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                model_output = call_model(model, batch, args, pass_batch)
                loss, _, pred_loss, _ = compute_total_loss(model_output, batch_y, criterion, args)
            scaler.scale(loss).backward()
            if getattr(args, "grad_clip", 0) and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            model_output = call_model(model, batch, args, pass_batch)
            loss, _, pred_loss, _ = compute_total_loss(model_output, batch_y, criterion, args)
            loss.backward()
            if getattr(args, "grad_clip", 0) and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        losses.append(loss.item())
        pred_losses.append(pred_loss.item())

    return float(np.mean(losses)), float(np.mean(pred_losses))


@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, device, args, pass_batch):
    model.eval()
    losses = []
    pred_losses = []

    for raw_batch in val_loader:
        batch = move_batch_to_device(raw_batch, device)
        batch_y = batch["y"]
        model_output = call_model(model, batch, args, pass_batch)
        loss, _, pred_loss, _ = compute_total_loss(model_output, batch_y, criterion, args)
        losses.append(loss.item())
        pred_losses.append(pred_loss.item())

    return float(np.mean(losses)), float(np.mean(pred_losses))


@torch.no_grad()
def test_model(model, test_loader, pv_scaler, device, checkpoint_path, target_dim, args, pass_batch):
    print("Loading best model for testing...")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    preds, trues, future_indices = [], [], []

    for raw_batch in test_loader:
        batch = move_batch_to_device(raw_batch, device)
        batch_y = batch["y"]
        model_output = call_model(model, batch, args, pass_batch)
        outputs, _ = parse_model_output(model_output)

        preds.append(outputs.detach().cpu().numpy())
        trues.append(batch_y.detach().cpu().numpy())

        if "index" not in batch:
            raise KeyError("test_model requires batch['index'] to save timestamp-aligned prediction CSV files.")
        starts = batch["index"].detach().cpu().numpy().astype(np.int64)
        horizon_offsets = np.arange(int(args.pred_len), dtype=np.int64)[None, :]
        target_start = starts[:, None] + int(args.seq_len)
        future_indices.append(target_start + horizon_offsets)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    future_indices = np.concatenate(future_indices, axis=0)

    preds_inv = inverse_pv_scale(preds, pv_scaler, target_dim=target_dim)
    trues_inv = inverse_pv_scale(trues, pv_scaler, target_dim=target_dim)

    return preds_inv, trues_inv, future_indices


@torch.no_grad()
def measure_inference_time(model, loader, device, args, pass_batch):
    model.eval()
    warmup = int(getattr(args, "timing_warmup_batches", 5))
    measure = int(getattr(args, "timing_measure_batches", 30))

    times = []
    total_samples = 0
    total_points = 0

    for i, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model_output = call_model(model, batch, args, pass_batch)
        outputs, _ = parse_model_output(model_output)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        if i >= warmup:
            elapsed_ms = (end - start) * 1000.0
            times.append(elapsed_ms)
            total_samples += batch["x"].shape[0]
            total_points += int(np.prod(outputs.shape))
            if len(times) >= measure:
                break

    if len(times) == 0:
        return {
            "forward_ms_mean": np.nan,
            "forward_ms_std": np.nan,
            "forward_ms_min": np.nan,
            "forward_ms_max": np.nan,
            "ms_per_sample": np.nan,
            "ms_per_output_point": np.nan,
            "measured_batches": 0,
        }

    times = np.array(times, dtype=np.float64)
    return {
        "forward_ms_mean": float(times.mean()),
        "forward_ms_std": float(times.std()),
        "forward_ms_min": float(times.min()),
        "forward_ms_max": float(times.max()),
        "ms_per_sample": float(times.sum() / max(total_samples, 1)),
        "ms_per_output_point": float(times.sum() / max(total_points, 1)),
        "measured_batches": int(len(times)),
        "total_samples": int(total_samples),
        "total_output_points": int(total_points),
    }


class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, path="checkpoint.pth", verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.inf
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def _prediction_csv_columns(target_dim: int):

    return [f"true_{i}" for i in range(target_dim)] + [f"pred_{i}" for i in range(target_dim)]


def _write_prediction_pair_csv(true_2d: np.ndarray, pred_2d: np.ndarray, save_path, target_dim: int):


    true_2d = np.asarray(true_2d)
    pred_2d = np.asarray(pred_2d)
    if true_2d.shape != pred_2d.shape:
        raise ValueError(f"true/pred shapes must match, got {true_2d.shape} and {pred_2d.shape}.")
    if true_2d.ndim != 2 or true_2d.shape[1] != target_dim:
        raise ValueError(f"Expected [num_rows, {target_dim}], got {true_2d.shape}.")

    columns = _prediction_csv_columns(target_dim)
    out = np.concatenate([true_2d, pred_2d], axis=1)
    df = pd.DataFrame(out, columns=columns)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def save_first_step_prediction_csv(preds, trues, save_path, target_dim):


    _write_prediction_pair_csv(
        true_2d=trues[:, 0, :],
        pred_2d=preds[:, 0, :],
        save_path=save_path,
        target_dim=target_dim,
    )


def save_overlap_average_prediction_csv(preds, trues, future_indices, save_path, target_dim):


    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)
    future_indices = np.asarray(future_indices, dtype=np.int64)

    if preds.shape != trues.shape:
        raise ValueError(f"preds/trues shapes must match, got {preds.shape} and {trues.shape}.")
    if preds.ndim != 3 or preds.shape[2] != target_dim:
        raise ValueError(f"Expected preds/trues shape [num_windows, pred_len, {target_dim}], got {preds.shape}.")
    if future_indices.shape != preds.shape[:2]:
        raise ValueError(
            f"future_indices shape {future_indices.shape} must match preds[:2] {preds.shape[:2]}."
        )

    min_idx = int(future_indices.min())
    max_idx = int(future_indices.max())
    num_times = max_idx - min_idx + 1

    pred_sum = np.zeros((num_times, target_dim), dtype=np.float64)
    true_sum = np.zeros((num_times, target_dim), dtype=np.float64)
    count = np.zeros((num_times, target_dim), dtype=np.float64)

    flat_offsets = (future_indices.reshape(-1) - min_idx).astype(np.int64)
    flat_pred = preds.reshape(-1, target_dim)
    flat_true = trues.reshape(-1, target_dim)

    np.add.at(pred_sum, flat_offsets, flat_pred)
    np.add.at(true_sum, flat_offsets, flat_true)
    np.add.at(count, flat_offsets, 1.0)

    covered = count[:, 0] > 0
    pred_avg = pred_sum[covered] / np.maximum(count[covered], 1.0)
    true_avg = true_sum[covered] / np.maximum(count[covered], 1.0)

    _write_prediction_pair_csv(
        true_2d=true_avg,
        pred_2d=pred_avg,
        save_path=save_path,
        target_dim=target_dim,
    )

    return {
        "rows": int(covered.sum()),
        "min_time_index": min_idx,
        "max_time_index": max_idx,
        "min_votes_per_timestamp": int(count[covered, 0].min()) if np.any(covered) else 0,
        "max_votes_per_timestamp": int(count[covered, 0].max()) if np.any(covered) else 0,
    }


def save_metrics_csv(metrics, timing_info, args, dataset_name, save_path, total_params, trainable_params, meta):
    scenario = meta.get("scenario", {})

    row = {
        "dataset": dataset_name,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "scenario_name": args.scenario_name,
        "impute_method": args.impute_method,
        "append_obs_mask": meta.get("append_obs_mask", None),
        "noise_std": scenario.get("noise_std", None),
        "missing_rate": scenario.get("missing_rate", None),
        "missing_pattern": scenario.get("missing_pattern", None),
        "aux_delay_steps": scenario.get("aux_delay_steps", None),
        "knn_n_neighbors": getattr(args, "knn_n_neighbors", None),
        "knn_weights": getattr(args, "knn_weights", None),
        "lambda_impute": float(getattr(args, "lambda_impute", 0.0)),
        "lambda_unc": float(getattr(args, "lambda_unc", 0.0)),
        "lambda_smooth": float(getattr(args, "lambda_smooth", 0.0)),
        "lambda_graph": float(getattr(args, "lambda_graph", 0.0)),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "enc_in": args.enc_in,
        "target_dim": args.target_dim,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "loss": args.loss,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_params_M": float(total_params / 1e6),
        "trainable_params_M": float(trainable_params / 1e6),
    }

    row.update(metrics)
    row.update(timing_info)

    df = pd.DataFrame([row])
    if os.path.exists(save_path):
        old = pd.read_csv(save_path)
        df = pd.concat([old, df], axis=0, ignore_index=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")


def apply_optional_overrides(args):
    if args.data_override is not None:
        args.data = args.data_override
    if args.batch_size_override is not None:
        args.batch_size = args.batch_size_override
    if args.epochs_override is not None:
        args.epochs = args.epochs_override
    if args.seq_len_override is not None:
        args.seq_len = args.seq_len_override
    if args.pred_len_override is not None:
        args.pred_len = args.pred_len_override
    if args.seed_override is not None:
        args.seed = args.seed_override
    if args.lr_override is not None:
        args.lr = args.lr_override
    if args.patience_override is not None:
        args.patience = args.patience_override
    if args.decay_patience_override is not None:
        args.decay_patience = args.decay_patience_override

    args.batchsize = args.batch_size
    args.enc_out = getattr(args, "enc_out", args.target_dim)
    return args


def resolve_impute_method(scenario_name: str, impute_method: str, model_type: str) -> str:


    if impute_method != "auto":
        return impute_method

    if model_type == "ours":
        return "zero"
    if scenario_name.startswith("missing") or scenario_name.startswith("mixed"):
        return "knn"
    return "forward_fill"


def apply_cli_overrides_after_config(cli_args, saved_cli):


    for key, value in saved_cli.items():
        if value is not None:
            setattr(cli_args, key, value)
    return cli_args


def print_batch_shapes(sample_batch):
    parts = []
    for key in ["x", "y", "pv_obs", "obs_mask", "pv_clean", "day_mask", "quality", "index"]:
        if key in sample_batch and isinstance(sample_batch[key], torch.Tensor):
            parts.append(f"{key}={tuple(sample_batch[key].shape)}")
    print("[Batch] " + ", ".join(parts))


def main(cli_args):
    saved_cli = {
        "scenario_name": cli_args.scenario_name,
        "impute_method": cli_args.impute_method,
        "model_type": cli_args.model_type,
        "knn_n_neighbors": cli_args.knn_n_neighbors,
        "knn_weights": cli_args.knn_weights,
    }

    config = get_config(cli_args.model_name)
    for key, value in vars(config).items():
        setattr(cli_args, key, value)
    args = apply_cli_overrides_after_config(cli_args, saved_cli)
    args = apply_optional_overrides(args)


    args.lambda_impute = float(getattr(args, "lambda_impute", 0.0) or 0.0)
    args.lambda_unc = float(getattr(args, "lambda_unc", 0.0) or 0.0)
    args.lambda_smooth = float(getattr(args, "lambda_smooth", 0.0) or 0.0)
    args.lambda_graph = float(getattr(args, "lambda_graph", 0.0) or 0.0)

    args.impute_method = resolve_impute_method(args.scenario_name, args.impute_method, args.model_type)
    if args.model_type == "ours" and args.impute_method != "zero":
        raise ValueError("For model_type='ours', impute_method must resolve to 'zero'.")

    fix_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_name = os.path.splitext(os.path.basename(args.data))[0]

    checkpoint_dir = Path(args.checkpoint_root) / dataset_name / args.scenario_name
    result_dir = Path(args.result_root) / dataset_name / args.scenario_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    run_identifier = (
        f"{args.model_name}_type-{args.model_type}_scenario-{args.scenario_name}_imp-{args.impute_method}_"
        f"len{args.seq_len}-{args.pred_len}_seed{args.seed}"
    )
    checkpoint_path = checkpoint_dir / f"{run_identifier}.pth"
    first_step_csv_path = result_dir / f"{run_identifier}_first_step.csv"
    overlap_avg_csv_path = result_dir / f"{run_identifier}_overlap_avg.csv"
    metrics_csv_path = result_dir / "metrics_summary.csv"

    print("=" * 100)
    print("[PV QUALITY FORECASTING] STARTING TRAINING EXPERIMENT")
    print("=" * 100)
    print(f"[Run] Model Name     : {args.model_name}")
    print(f"[Run] Model Type     : {args.model_type}")
    print(f"[Run] Dataset Name   : {dataset_name}")
    print(f"[Run] Data Path      : {args.data}")
    print(f"[Run] Scenario       : {args.scenario_name}")
    print(f"[Run] Imputation     : {args.impute_method}")
    print(f"[Run] Window         : seq_len={args.seq_len} -> pred_len={args.pred_len}")
    print(f"[Run] Config dims    : enc_in={args.enc_in}, target_dim={args.target_dim}")
    print(f"[Run] Hyperparams    : lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"[Run] Aux loss lambdas: impute={args.lambda_impute}, unc={args.lambda_unc}, "
          f"smooth={args.lambda_smooth}, graph={args.lambda_graph}")
    print(f"[Run] Device         : {device}, seed={args.seed}")
    print(f"[Run] Checkpoint     : {checkpoint_path}")
    print("=" * 100)

    print("\n-----------------------------------Loading quality-aware data-----------------------------------")

    train_ratio = args.split_ratio[0] if hasattr(args, "split_ratio") else 0.8
    val_ratio = args.split_ratio[1] if hasattr(args, "split_ratio") else 0.1

    data_cfg = PVDataConfig(
        csv_path=args.data,
        num_sites=args.target_dim,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        scenario_name=args.scenario_name,
        model_type=args.model_type,
        impute_method=args.impute_method,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=args.drop_last,
        day_threshold=args.day_threshold,
        scale=args.scale,
        knn_n_neighbors=args.knn_n_neighbors,
        knn_weights=args.knn_weights,
    )

    train_loader, val_loader, test_loader, meta = build_pv_quality_dataloaders(data_cfg)

    print(f"[Data] model_type={args.model_type}, impute_method={args.impute_method}, "
          f"append_obs_mask={meta.get('append_obs_mask', False)}")
    print(f"[Data] input_dim={meta['input_dim']} | target_dim={meta['target_dim']} | "
          f"aux_dim={meta['aux_dim']} | aux_per_site={meta.get('aux_per_site', 'NA')}")
    print(f"[Data] knn_imputer_is_none={meta.get('knn_imputer') is None}")
    print(f"[Data] scenario={meta['scenario']}")
    print(f"[Data] split: train_end={meta['train_end']} | val_end={meta['val_end']} | total_len={meta['total_len']}")
    print(f"[Data] windows: train={len(train_loader.dataset)} | val={len(val_loader.dataset)} | test={len(test_loader.dataset)}")

    sample_batch = next(iter(train_loader))
    print_batch_shapes(sample_batch)


    if args.model_type == "ours":
        args.enc_in = int(meta["input_dim"])
        args.enc_out = int(meta["target_dim"])
    elif meta["input_dim"] != args.enc_in:
        raise ValueError(
            f"Model enc_in={args.enc_in}, but dataset input_dim={meta['input_dim']}. "
            f"Please check your config."
        )
    if meta["target_dim"] != args.target_dim:
        raise ValueError(
            f"Model target_dim={args.target_dim}, but dataset target_dim={meta['target_dim']}. "
            f"Please check your config."
        )

    model = load_model(args).to(device)
    pass_batch = (args.model_type == "ours") and model_accepts_batch(model)
    total_params, trainable_params = count_parameters(model)

    print("\n-----------------------------------Model Parameter Summary-----------------------------------")
    print(f"[Model] forward_accepts_batch={pass_batch}")
    print(f"[Model] Total Params     : {total_params:,}")
    print(f"[Model] Trainable Params : {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = get_criterion(args.loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.decay_patience,
        verbose=True,
        min_lr=1e-6,
    )
    early_stopping = EarlyStopping(patience=args.patience, path=str(checkpoint_path), verbose=True)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(bool(args.use_amp) and device.type == "cuda"))

    print("\n-----------------------------------Start Training----------------------------------")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_pred_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args, pass_batch, scaler=amp_scaler
        )
        val_loss, val_pred_loss = validate_one_epoch(model, val_loader, criterion, device, args, pass_batch)
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch: {epoch + 1:03d} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | Train Pred: {train_pred_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val Pred: {val_pred_loss:.6f} | LR: {current_lr:.6e}"
        )

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\n-----------------------------------Start Testing-----------------------------------")
    preds, trues, future_indices = test_model(
        model,
        test_loader,
        meta["pv_scaler"],
        device,
        checkpoint_path=str(checkpoint_path),
        target_dim=args.target_dim,
        args=args,
        pass_batch=pass_batch,
    )

    metrics = compute_metrics(preds, trues, day_threshold=args.day_threshold)
    imputation_metrics = evaluate_imputation_metrics(
        model=model,
        test_loader=test_loader,
        pv_scaler=meta["pv_scaler"],
        device=device,
        target_dim=args.target_dim,
        args=args,
        pass_batch=pass_batch,
    )
    metrics.update(imputation_metrics)
    timing_info = measure_inference_time(model, test_loader, device, args, pass_batch)

    save_first_step_prediction_csv(
        preds,
        trues,
        first_step_csv_path,
        target_dim=args.target_dim,
    )
    overlap_save_info = save_overlap_average_prediction_csv(
        preds,
        trues,
        future_indices,
        overlap_avg_csv_path,
        target_dim=args.target_dim,
    )
    save_metrics_csv(
        metrics,
        timing_info,
        args,
        dataset_name,
        metrics_csv_path,
        total_params=total_params,
        trainable_params=trainable_params,
        meta=meta,
    )

    print("\n" + "=" * 100)
    print("[PV QUALITY FORECASTING] TRAINING & TESTING COMPLETED")
    print("=" * 100)
    print(f"[Metrics] Overall: MAE={metrics['MAE']:.4f}, WAPE={metrics['WAPE']:.4f}%, "
          f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    print(f"[Metrics] Daytime: MAE={metrics['Day_MAE']:.4f}, WAPE={metrics['Day_WAPE']:.4f}%, "
          f"RMSE={metrics['Day_RMSE']:.4f}, R2={metrics['Day_R2']:.4f}, Count={metrics['Day_Count']}")
    if int(metrics.get("Impute_Count", 0) or 0) > 0 and not np.isnan(metrics.get("Impute_MAE", np.nan)):
        print(f"[Metrics] Impute : MAE={metrics['Impute_MAE']:.4f}, "
              f"RMSE={metrics['Impute_RMSE']:.4f}, Count={metrics['Impute_Count']}, "
              f"Source={metrics.get('Impute_Source', 'NA')}, Space=raw")
    else:
        print(f"[Metrics] Impute : unavailable or no missing points. "
              f"Source={metrics.get('Impute_Source', 'NA')}, Count={metrics.get('Impute_Count', 0)}")
    print(f"[Timing] Forward : {timing_info['forward_ms_mean']:.4f} +/- {timing_info['forward_ms_std']:.4f} ms "
          f"(min={timing_info['forward_ms_min']:.4f}, max={timing_info['forward_ms_max']:.4f})")
    print(f"[Timing] Sample  : {timing_info['ms_per_sample']:.6f} ms/sample")
    print(f"[Timing] Point   : {timing_info['ms_per_output_point']:.9f} ms/output point")
    print(f"[Saved] Best Model           : {checkpoint_path}")
    print(f"[Saved] First-step CSV       : {first_step_csv_path}")
    print(f"[Saved] Overlap-average CSV  : {overlap_avg_csv_path}")
    print(f"[Saved] Overlap rows         : {overlap_save_info['rows']} "
          f"| time_index={overlap_save_info['min_time_index']}..{overlap_save_info['max_time_index']} "
          f"| votes={overlap_save_info['min_votes_per_timestamp']}..{overlap_save_info['max_votes_per_timestamp']}")
    print(f"[Saved] Metrics Summary      : {metrics_csv_path}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PV forecasting under heterogeneous measurement quality")


    parser.add_argument(
        "--scenario_name",
        type=str,
        default=None,
        choices=[
            "clean", "noise_01", "noise_02", "noise_03",
            "missing_10", "missing_30", "missing_50",
            "mixed_10", "mixed_20", "mixed_30",
        ],
        help="PV data quality scenario.",
    )
    parser.add_argument(
        "--impute_method",
        type=str,
        default=None,
        choices=["auto", "forward_fill", "zero", "train_mean", "knn"],
        help="Missing-value method. auto: ours->zero, baseline missing/mixed->knn, baseline clean/noise->forward_fill.",
    )
    parser.add_argument("--model_type", type=str, default=None, choices=["baseline", "ours"])
    parser.add_argument("--knn_n_neighbors", type=int, default=None)
    parser.add_argument("--knn_weights", type=str, default=None, choices=["uniform", "distance"])


    parser.add_argument("--model_name", type=str, default="DLinear", help="Model name under ./models and config name.")


    parser.add_argument("--data_override", type=str, default=None, help="Optional data path override.")
    parser.add_argument("--batch_size_override", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--epochs_override", type=int, default=None, help="Optional epochs override.")
    parser.add_argument("--seq_len_override", type=int, default=None, help="Optional seq_len override.")
    parser.add_argument("--pred_len_override", type=int, default=None, help="Optional pred_len override.")
    parser.add_argument("--seed_override", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--lr_override", type=float, default=None, help="Optional learning-rate override.")
    parser.add_argument("--patience_override", type=int, default=None, help="Optional early-stopping patience override.")
    parser.add_argument("--decay_patience_override", type=int, default=None, help="Optional LR scheduler patience override.")
    parser.add_argument("--drop_last", action="store_true", help="Force drop_last=True for training DataLoader.")


    parser.add_argument("--lambda_impute", type=float, default=None)
    parser.add_argument("--lambda_unc", type=float, default=None)
    parser.add_argument("--lambda_smooth", type=float, default=None)
    parser.add_argument("--lambda_graph", type=float, default=None)

    cli_args = parser.parse_args()
    main(cli_args)
