import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from sklearn.impute import KNNImputer


AUX_DELAY_STEPS = 4         # one-hour delay for 15-minute resolution auxiliary features


# ============================================================
# Scenario definitions
# ------------------------------------------------------------
# Principle:
#   1. The original CSV is always treated as clean ground truth.
#   2. The scenario only corrupts the historical input x.
#   3. The prediction target y is always the clean future PV power.
#   4. PV primary features are never time-delayed.
#   5. Meteorological auxiliary features are always delayed by 4 steps,
#      i.e., one hour for 15-minute resolution data.
# ============================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "clean": {
        "noise_std": 0.0,
        "missing_rate": 0.0,
        "missing_pattern": "none",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "noise_01": {
        "noise_std": 0.1,
        "missing_rate": 0.0,
        "missing_pattern": "none",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "noise_02": {
        "noise_std": 0.2,
        "missing_rate": 0.0,
        "missing_pattern": "none",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "noise_03": {
        "noise_std": 0.3,
        "missing_rate": 0.0,
        "missing_pattern": "none",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "missing_10": {
        "noise_std": 0.0,
        "missing_rate": 0.1,
        "missing_pattern": "random_point",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "missing_30": {
        "noise_std": 0.0,
        "missing_rate": 0.3,
        "missing_pattern": "random_point",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "missing_50": {
        "noise_std": 0.0,
        "missing_rate": 0.5,
        "missing_pattern": "random_point",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "mixed_10": {
        "noise_std": 0.1,
        "missing_rate": 0.1,
        "missing_pattern": "random_point",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "mixed_20": {
        "noise_std": 0.2,
        "missing_rate": 0.2,
        "missing_pattern": "random_point",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
    "mixed_30": {
        "noise_std": 0.3,
        "missing_rate": 0.3,
        "missing_pattern": "random_point",
        "aux_delay_steps": AUX_DELAY_STEPS,
    },
}


@dataclass
class PVDataConfig:
    csv_path: str
    num_sites: int
    seq_len: int
    pred_len: int
    batch_size: int
    train_ratio: float
    val_ratio: float
    scenario_name: str
    model_type: str
    impute_method: str
    seed: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    day_threshold: float
    scale: bool

    # KNN imputation settings.
    # Only used when impute_method == "knn".
    knn_n_neighbors: int = 5
    knn_weights: str = "distance"

    # Layout of raw auxiliary columns in the CSV.
    # feature_major: aux_0_site_0..aux_0_site_N, aux_1_site_0..aux_1_site_N, ...
    # unit_major:    site_0_aux_0..site_0_aux_K, site_1_aux_0..site_1_aux_K, ...
    # The dataloader always converts auxiliary inputs to unit_major before
    # concatenating them into x, so models can reshape aux as [B, L, N, K].
    aux_layout: str = "feature_major"


class StandardScaler:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std < 1e-6] = 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return data * self.std + self.mean


class IdentityScaler:
    def fit(self, data: np.ndarray):
        return None

    def transform(self, data: np.ndarray) -> np.ndarray:
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data


def get_scenario(scenario_name: str) -> Dict[str, Any]:
    if scenario_name not in SCENARIOS:
        valid_names = ", ".join(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario_name={scenario_name}. Valid scenarios: {valid_names}")
    return dict(SCENARIOS[scenario_name])


def time_delay(data: np.ndarray, delay_steps: int) -> np.ndarray:
    """
    Apply causal time delay:
        delayed[t] = data[t - delay_steps]

    For the first delay_steps samples, the earliest available value is used.
    This avoids using future information.
    """
    if delay_steps <= 0:
        return data.copy()

    delayed = np.empty_like(data)
    delayed[:delay_steps] = data[0:1]
    delayed[delay_steps:] = data[:-delay_steps]
    return delayed.astype(np.float32)


def add_daytime_gaussian_noise(
    pv_obs_scaled: np.ndarray,
    pv_obs_raw: np.ndarray,
    noise_std: float,
    day_threshold: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise only to daytime PV values.

    The noise is added in the scaled space. Daytime is judged using raw PV power:
        day_mask = pv_obs_raw > day_threshold

    Args:
        pv_obs_scaled: observed PV input without power delay, shape [T, N]
        pv_obs_raw: raw PV input without power delay, shape [T, N]
        noise_std: std of Gaussian noise in scaled space
        day_threshold: raw PV value threshold for daytime
        rng: numpy random generator

    Returns:
        pv_noisy_scaled: noisy PV input, shape [T, N]
        day_mask: daytime mask, shape [T, N], 1 for daytime and 0 for nighttime
    """
    day_mask = (pv_obs_raw > day_threshold).astype(np.float32)

    if noise_std <= 0:
        return pv_obs_scaled.astype(np.float32), day_mask

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=pv_obs_scaled.shape,
    ).astype(np.float32)

    pv_noisy_scaled = pv_obs_scaled + noise * day_mask
    return pv_noisy_scaled.astype(np.float32), day_mask


def generate_missing_mask(
    shape: Tuple[int, int],
    missing_rate: float,
    missing_pattern: str,
    rng: np.random.Generator,
    block_len: int = 8,
) -> np.ndarray:
    """
    Generate observed mask for PV power.

    Returns:
        obs_mask: shape [T, N], 1 for observed and 0 for missing.
    """
    T, N = shape

    if missing_rate <= 0 or missing_pattern == "none":
        return np.ones((T, N), dtype=np.float32)

    if not 0.0 <= missing_rate < 1.0:
        raise ValueError(f"missing_rate should be in [0, 1), got {missing_rate}")

    if missing_pattern == "random_point":
        missing = rng.random((T, N)) < missing_rate
        return (~missing).astype(np.float32)

    if missing_pattern == "block":
        missing = np.zeros((T, N), dtype=bool)
        target_missing = int(T * N * missing_rate)
        current_missing = 0

        while current_missing < target_missing:
            site = int(rng.integers(0, N))
            start = int(rng.integers(0, max(T - block_len + 1, 1)))
            end = min(start + block_len, T)
            before = missing[:, site].sum()
            missing[start:end, site] = True
            after = missing[:, site].sum()
            current_missing += int(after - before)

        return (~missing).astype(np.float32)

    if missing_pattern == "station":
        missing = np.zeros((T, N), dtype=bool)
        num_missing_sites = max(1, int(N * missing_rate))
        sites = rng.choice(N, size=num_missing_sites, replace=False)
        missing[:, sites] = True
        return (~missing).astype(np.float32)

    raise ValueError(
        f"Unknown missing_pattern={missing_pattern}. "
        f"Choose from: none, random_point, block, station."
    )


def apply_missing(pv_obs_scaled: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    pv_missing = pv_obs_scaled.copy()
    pv_missing[obs_mask < 0.5] = np.nan
    return pv_missing.astype(np.float32)


def causal_forward_fill(data: np.ndarray, initial_values: np.ndarray) -> np.ndarray:
    """
    Causal forward filling for missing values.

    It never uses future values. If the first value is missing, it uses initial_values.
    This is safer than pandas ffill().bfill() for time-series experiments.
    """
    filled = data.copy().astype(np.float32)
    last = initial_values.astype(np.float32).copy()

    for t in range(filled.shape[0]):
        row = filled[t]
        nan_pos = np.isnan(row)
        row[nan_pos] = last[nan_pos]
        last = row.copy()
        filled[t] = row

    return filled.astype(np.float32)


def knn_impute_with_train_fit(
    data: np.ndarray,
    train_end: int,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> Tuple[np.ndarray, Optional[Any]]:
    """
    KNN imputation for PV missing values.

    Important:
        The KNN imputer is fitted only on the training split to avoid data leakage.
        Then it transforms the full sequence using the train-fitted imputer.

    Args:
        data:
            PV observation with NaNs, shape [T, num_sites].
        train_end:
            End index of training split.
        n_neighbors:
            Number of nearest neighbors.
        weights:
            'uniform' or 'distance'.

    Returns:
        imputed_data:
            KNN-imputed PV observation, shape [T, num_sites].
        imputer:
            Fitted KNNImputer, or None if no NaN exists.
    """
    if not np.isnan(data).any():
        return data.astype(np.float32), None

    if KNNImputer is None:
        raise ImportError(
            "scikit-learn is required for impute_method='knn'. "
            "Please install it with: pip install scikit-learn"
        )

    if train_end <= 0 or train_end > data.shape[0]:
        raise ValueError(f"Invalid train_end={train_end} for data length {data.shape[0]}.")

    train_data = data[:train_end].astype(np.float32)

    # Fallback for extreme cases where an entire PV station is missing in training.
    train_col_mean = np.nanmean(train_data, axis=0).astype(np.float32)
    train_col_mean = np.nan_to_num(train_col_mean, nan=0.0).astype(np.float32)

    safe_data = data.copy().astype(np.float32)
    empty_cols = np.isnan(train_data).all(axis=0)
    if empty_cols.any():
        safe_data[:, empty_cols] = train_col_mean[empty_cols]

    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="nan_euclidean",
    )

    # Fit only on training split.
    imputer.fit(safe_data[:train_end])

    # Transform the whole sequence using the train-fitted imputer.
    imputed_data = imputer.transform(safe_data).astype(np.float32)

    # Final safety check.
    imputed_data = np.nan_to_num(imputed_data, nan=0.0).astype(np.float32)

    return imputed_data, imputer


def fill_missing_values(
    data: np.ndarray,
    method: str,
    train_initial_values: np.ndarray,
) -> np.ndarray:
    """
    Fill NaN values for conventional baselines or for numerical model input.

    Supported non-KNN methods:
        - forward_fill: causal forward filling
        - zero: replace NaN with 0
        - train_mean: replace NaN with training mean values

    KNN is handled separately by knn_impute_with_train_fit(), because it needs train_end.
    """
    if not np.isnan(data).any():
        return data.astype(np.float32)

    if method == "forward_fill":
        return causal_forward_fill(data, train_initial_values)

    if method == "zero":
        return np.nan_to_num(data, nan=0.0).astype(np.float32)

    if method == "train_mean":
        filled = data.copy().astype(np.float32)
        nan_pos = np.isnan(filled)
        filled[nan_pos] = np.take(train_initial_values, np.where(nan_pos)[1])
        return filled.astype(np.float32)

    raise ValueError(
        f"Unknown impute_method={method}. "
        f"Choose from: forward_fill, zero, train_mean, knn."
    )


def validate_config(cfg: PVDataConfig):
    if not 0.0 < cfg.train_ratio < 1.0:
        raise ValueError(f"train_ratio should be in (0, 1), got {cfg.train_ratio}")

    if not 0.0 <= cfg.val_ratio < 1.0:
        raise ValueError(f"val_ratio should be in [0, 1), got {cfg.val_ratio}")

    if cfg.train_ratio + cfg.val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio should be < 1, got "
            f"{cfg.train_ratio + cfg.val_ratio}"
        )

    if cfg.model_type not in {"baseline", "ours"}:
        raise ValueError("model_type should be either 'baseline' or 'ours'.")

    valid_impute_methods = {"forward_fill", "zero", "train_mean", "knn"}
    if cfg.impute_method not in valid_impute_methods:
        raise ValueError(
            f"impute_method should be one of {valid_impute_methods}, got {cfg.impute_method}."
        )

    # For ours/proposed models, do not use KNN and keep missing-value placeholders as zero-filled
    # PV values paired with explicit observation masks.
    if cfg.model_type == "ours" and cfg.impute_method != "zero":
        raise ValueError(
            "model_type='ours' requires impute_method='zero'. "
            "KNN/forward_fill/train_mean are not allowed for ours."
        )

    if cfg.aux_layout not in {"feature_major", "unit_major"}:
        raise ValueError(
            f"aux_layout should be either 'feature_major' or 'unit_major', got {cfg.aux_layout}."
        )

    if cfg.knn_n_neighbors <= 0:
        raise ValueError("knn_n_neighbors should be a positive integer.")

    if cfg.knn_weights not in {"uniform", "distance"}:
        raise ValueError("knn_weights should be either 'uniform' or 'distance'.")

    if cfg.seq_len <= 0 or cfg.pred_len <= 0:
        raise ValueError("seq_len and pred_len should be positive integers.")


def split_power_aux(all_data: np.ndarray, num_sites: int) -> Tuple[np.ndarray, np.ndarray]:
    if all_data.shape[1] <= num_sites:
        raise ValueError(
            f"The CSV has {all_data.shape[1]} columns, which is not enough for "
            f"num_sites={num_sites}. Expected power columns plus auxiliary columns."
        )

    pv_raw = all_data[:, :num_sites]
    aux_raw = all_data[:, num_sites:]
    return pv_raw.astype(np.float32), aux_raw.astype(np.float32)


def reorder_auxiliary_features(
    aux_raw: np.ndarray,
    num_sites: int,
    aux_layout: str,
    aux_columns: Optional[list] = None,
) -> Tuple[np.ndarray, Optional[list], int]:
    """
    Convert raw auxiliary columns to unit-major order for model input.

    Supported raw layouts:
        feature_major:
            [aux_0_site_0 ... aux_0_site_N, aux_1_site_0 ... aux_1_site_N, ...]
        unit_major:
            [site_0_aux_0 ... site_0_aux_K, site_1_aux_0 ... site_1_aux_K, ...]

    Returned aux is always unit-major:
        [site_0_aux_0 ... site_0_aux_K, site_1_aux_0 ... site_1_aux_K, ...]

    Therefore, models can safely reshape the auxiliary part as [B, L, N, K].
    """
    aux_raw = aux_raw.astype(np.float32)
    total_len, aux_dim = aux_raw.shape

    if aux_dim % num_sites != 0:
        raise ValueError(
            f"Auxiliary dimension {aux_dim} is not divisible by num_sites={num_sites}. "
            "Expected auxiliary columns to be either feature-major "
            "[aux_k_site_i] or unit-major [site_i_aux_k]."
        )

    aux_per_site = aux_dim // num_sites

    if aux_columns is not None:
        aux_columns = list(aux_columns)
        if len(aux_columns) != aux_dim:
            raise ValueError(
                f"len(aux_columns)={len(aux_columns)} does not match aux_dim={aux_dim}."
            )

    if aux_layout == "feature_major":
        aux_model = (
            aux_raw.reshape(total_len, aux_per_site, num_sites)
            .transpose(0, 2, 1)
            .reshape(total_len, aux_dim)
            .astype(np.float32)
        )
        if aux_columns is None:
            aux_columns_model = None
        else:
            aux_columns_model = (
                np.asarray(aux_columns, dtype=object)
                .reshape(aux_per_site, num_sites)
                .T
                .reshape(-1)
                .tolist()
            )
        return aux_model, aux_columns_model, aux_per_site

    if aux_layout == "unit_major":
        return aux_raw.astype(np.float32), aux_columns, aux_per_site

    raise ValueError(
        f"Unknown aux_layout={aux_layout}. Choose from 'feature_major' or 'unit_major'."
    )


class PVQualityDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        y_clean: np.ndarray,
        pv_obs_input: np.ndarray,
        pv_clean_raw: np.ndarray,
        obs_mask_full: np.ndarray,
        day_mask_full: np.ndarray,
        split_start: int,
        split_end: int,
        seq_len: int,
        pred_len: int,
        num_sites: int,
        scenario_name: str,
        scenario: Dict[str, Any],
        model_type: str,
    ):
        self.x_full = x_full.astype(np.float32)
        self.y_clean = y_clean.astype(np.float32)
        self.pv_obs_input = pv_obs_input.astype(np.float32)
        self.pv_clean_raw = pv_clean_raw.astype(np.float32)
        self.obs_mask_full = obs_mask_full.astype(np.float32)
        self.day_mask_full = day_mask_full.astype(np.float32)

        self.split_start = int(split_start)
        self.split_end = int(split_end)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.num_sites = int(num_sites)
        self.scenario_name = scenario_name
        self.scenario = scenario
        self.model_type = model_type

        last_start = self.split_end - self.seq_len - self.pred_len + 1
        if last_start <= self.split_start:
            raise ValueError(
                f"Split too short for windowing. split_start={split_start}, "
                f"split_end={split_end}, seq_len={seq_len}, pred_len={pred_len}"
            )

        self.indices = np.arange(self.split_start, last_start, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self.indices[idx])
        e = s + self.seq_len
        y_s = e
        y_e = e + self.pred_len

        x = self.x_full[s:e]
        y = self.y_clean[y_s:y_e]

        pv_obs = self.pv_obs_input[s:e]
        obs_mask = self.obs_mask_full[s:e]
        day_mask = self.day_mask_full[s:e]
        y_raw = self.pv_clean_raw[y_s:y_e]
        pv_clean_hist = self.y_clean[s:e]

        quality = np.array(
            [
                float(self.scenario["noise_std"]),
                float(self.scenario["missing_rate"]),
                float(self.scenario["aux_delay_steps"]),
            ],
            dtype=np.float32,
        )

        batch = {
            "x": torch.from_numpy(x),                    # baseline: [seq_len, num_sites + aux_dim], ours: [seq_len, num_sites + num_sites + aux_dim]
            "y": torch.from_numpy(y),                    # [pred_len, num_sites]
            "pv_obs": torch.from_numpy(pv_obs),          # [seq_len, num_sites]
            "obs_mask": torch.from_numpy(obs_mask),      # [seq_len, num_sites]
            "pv_clean": torch.from_numpy(pv_clean_hist), # [seq_len, num_sites], clean historical PV (scaled)
            "day_mask": torch.from_numpy(day_mask),      # [seq_len, num_sites]
            "quality": torch.from_numpy(quality),        # [3]: [noise_std, missing_rate, aux_delay_steps]
            "y_raw": torch.from_numpy(y_raw),            # [pred_len, num_sites]
            "index": torch.tensor(s, dtype=torch.long),
        }

        return batch


def build_pv_quality_dataloaders(
    cfg: PVDataConfig,
    scenario_override: Optional[Dict[str, Any]] = None,
):
    """
    Build train/val/test dataloaders for PV quality scenarios.

    Data assumption:
        - The first cfg.num_sites columns are clean PV power.
        - The remaining columns are auxiliary features.
        - For your current dataset: 227 PV columns + 7 * 227 auxiliary columns.
        - Your raw CSV auxiliary columns are feature-major by default:
          aux_0_site_0 ... aux_0_site_226, aux_1_site_0 ... aux_1_site_226, ...
        - This dataloader converts auxiliary features to unit-major order before
          concatenating them into x:
          site_0_aux_0 ... site_0_aux_6, site_1_aux_0 ... site_1_aux_6, ...
          Therefore model code can reshape the auxiliary part as [B, L, N, 7].

    Output convention:
        x: historical corrupted input, shape [B, seq_len, input_dim]
        y: clean future PV target, shape [B, pred_len, num_sites]

    For missing-data scenarios:
        - PV observations are first corrupted by missing masks.
        - If impute_method == "knn", KNNImputer is fitted only on the training split.
        - The fitted imputer then transforms train/val/test to avoid data leakage.

    PV primary features are not time-delayed. Auxiliary features are always
    causally delayed by AUX_DELAY_STEPS, which corresponds to one hour for
    15-minute resolution data.
    """
    validate_config(cfg)

    scenario = get_scenario(cfg.scenario_name)
    if scenario_override is not None:
        scenario.update(scenario_override)

    # PV primary features are never delayed. Auxiliary features are fixed to a
    # one-hour delay. Keep this assignment after scenario_override so that no
    # external scenario can silently change the auxiliary delay setting.
    scenario["aux_delay_steps"] = AUX_DELAY_STEPS

    rng = np.random.default_rng(cfg.seed)

    df = pd.read_csv(cfg.csv_path)
    all_data = df.values.astype(np.float32)
    total_len = all_data.shape[0]

    pv_raw, aux_raw = split_power_aux(all_data, cfg.num_sites)
    raw_aux_columns = df.columns[cfg.num_sites:].tolist()
    aux_raw, aux_columns_model, aux_per_site = reorder_auxiliary_features(
        aux_raw=aux_raw,
        num_sites=cfg.num_sites,
        aux_layout=cfg.aux_layout,
        aux_columns=raw_aux_columns,
    )
    aux_dim = aux_raw.shape[1]

    train_end = int(total_len * cfg.train_ratio)
    val_end = int(total_len * (cfg.train_ratio + cfg.val_ratio))

    if train_end <= cfg.seq_len + cfg.pred_len:
        raise ValueError("Training split is too short for the given seq_len and pred_len.")

    if cfg.scale:
        pv_scaler = StandardScaler()
        aux_scaler = StandardScaler()
    else:
        pv_scaler = IdentityScaler()
        aux_scaler = IdentityScaler()

    # Fit scalers only on training split.
    pv_scaler.fit(pv_raw[:train_end])
    aux_scaler.fit(aux_raw[:train_end])

    pv_scaled = pv_scaler.transform(pv_raw).astype(np.float32)
    aux_scaled = aux_scaler.transform(aux_raw).astype(np.float32)

    # ------------------------------------------------------------
    # 1. PV primary observation.
    #    The main PV feature is not delayed. Only noise and missingness are
    #    applied to the historical PV input.
    # ------------------------------------------------------------
    pv_obs_scaled = pv_scaled.copy().astype(np.float32)
    pv_obs_raw = pv_raw.copy().astype(np.float32)

    # ------------------------------------------------------------
    # 2. Daytime-only noise on PV observation.
    #    Noise is added in scaled space.
    # ------------------------------------------------------------
    noise_std = float(scenario["noise_std"])
    pv_obs_scaled, day_mask = add_daytime_gaussian_noise(
        pv_obs_scaled=pv_obs_scaled,
        pv_obs_raw=pv_obs_raw,
        noise_std=noise_std,
        day_threshold=cfg.day_threshold,
        rng=rng,
    )

    # ------------------------------------------------------------
    # 3. Missing PV observation.
    #    For clean/noise scenarios, obs_mask is all ones.
    # ------------------------------------------------------------
    obs_mask = generate_missing_mask(
        shape=pv_obs_scaled.shape,
        missing_rate=float(scenario["missing_rate"]),
        missing_pattern=str(scenario["missing_pattern"]),
        rng=rng,
    )
    pv_obs_with_nan = apply_missing(pv_obs_scaled, obs_mask)

    # ------------------------------------------------------------
    # 4. Fill missing PV input.
    #    baseline:
    #       missing/mixed can use KNN (fit on train split only), or other methods.
    #    ours/proposed model:
    #       only zero fill is allowed; explicit obs_mask is appended into x.
    # ------------------------------------------------------------
    train_initial_values = np.nanmean(pv_obs_with_nan[:train_end], axis=0).astype(np.float32)
    train_initial_values = np.nan_to_num(train_initial_values, nan=0.0).astype(np.float32)

    knn_imputer = None
    if cfg.impute_method == "knn":
        pv_model_input, knn_imputer = knn_impute_with_train_fit(
            data=pv_obs_with_nan,
            train_end=train_end,
            n_neighbors=cfg.knn_n_neighbors,
            weights=cfg.knn_weights,
        )
    else:
        pv_model_input = fill_missing_values(
            data=pv_obs_with_nan,
            method=cfg.impute_method,
            train_initial_values=train_initial_values,
        )

    # ------------------------------------------------------------
    # 5. Auxiliary features are always delayed by 4 steps.
    #    They are not sparse and not missing in your current setting.
    # ------------------------------------------------------------
    aux_delay_steps = int(scenario["aux_delay_steps"])
    aux_model_input = time_delay(aux_scaled, aux_delay_steps)

    # ------------------------------------------------------------
    # 6. Final model input and clean target.
    #    baseline x: [pv_model_input, aux_model_input]
    #    ours     x: [pv_model_input, obs_mask, aux_model_input]
    # ------------------------------------------------------------
    append_obs_mask = (cfg.model_type == "ours")
    if append_obs_mask:
        x_full = np.concatenate([pv_model_input, obs_mask, aux_model_input], axis=1).astype(np.float32)
    else:
        x_full = np.concatenate([pv_model_input, aux_model_input], axis=1).astype(np.float32)
    y_clean = pv_scaled.astype(np.float32)

    train_dataset = PVQualityDataset(
        x_full=x_full,
        y_clean=y_clean,
        pv_obs_input=pv_model_input,
        pv_clean_raw=pv_raw,
        obs_mask_full=obs_mask,
        day_mask_full=day_mask,
        split_start=0,
        split_end=train_end,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        num_sites=cfg.num_sites,
        scenario_name=cfg.scenario_name,
        scenario=scenario,
        model_type=cfg.model_type,
    )

    val_dataset = PVQualityDataset(
        x_full=x_full,
        y_clean=y_clean,
        pv_obs_input=pv_model_input,
        pv_clean_raw=pv_raw,
        obs_mask_full=obs_mask,
        day_mask_full=day_mask,
        split_start=train_end,
        split_end=val_end,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        num_sites=cfg.num_sites,
        scenario_name=cfg.scenario_name,
        scenario=scenario,
        model_type=cfg.model_type,
    )

    test_dataset = PVQualityDataset(
        x_full=x_full,
        y_clean=y_clean,
        pv_obs_input=pv_model_input,
        pv_clean_raw=pv_raw,
        obs_mask_full=obs_mask,
        day_mask_full=day_mask,
        split_start=val_end,
        split_end=total_len,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        num_sites=cfg.num_sites,
        scenario_name=cfg.scenario_name,
        scenario=scenario,
        model_type=cfg.model_type,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    meta = {
        "scenario_name": cfg.scenario_name,
        "scenario": scenario,
        "model_type": cfg.model_type,
        "impute_method": cfg.impute_method,
        "knn_n_neighbors": cfg.knn_n_neighbors,
        "knn_weights": cfg.knn_weights,
        "num_sites": cfg.num_sites,
        "aux_dim": aux_dim,
        "aux_per_site": aux_per_site,
        "aux_layout_original": cfg.aux_layout,
        "aux_layout_model": "unit_major",
        "input_dim": x_full.shape[1],
        "target_dim": cfg.num_sites,
        "append_obs_mask": append_obs_mask,
        "total_len": total_len,
        "train_end": train_end,
        "val_end": val_end,
        "pv_columns": df.columns[:cfg.num_sites].tolist(),
        "aux_columns_raw": raw_aux_columns,
        "aux_columns": aux_columns_model,
        "pv_scaler": pv_scaler,
        "aux_scaler": aux_scaler,
        "knn_imputer": knn_imputer,
    }

    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    cfg = PVDataConfig(
        csv_path="./pv_power_selected_aux_7features.csv",
        num_sites=227,
        seq_len=96,
        pred_len=4,
        batch_size=64,
        train_ratio=0.8,
        val_ratio=0.1,
        scenario_name="missing_30",
        model_type="baseline",
        impute_method="knn",
        seed=2026,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        day_threshold=1e-6,
        scale=True,
        knn_n_neighbors=5,
        knn_weights="distance",
        aux_layout="feature_major",
    )

    train_loader, val_loader, test_loader, meta = build_pv_quality_dataloaders(cfg)

    print("Meta:")
    for key, value in meta.items():
        if key not in {
            "pv_scaler",
            "aux_scaler",
            "pv_columns",
            "aux_columns",
            "knn_imputer",
        }:
            print(f"  {key}: {value}")

    batch = next(iter(train_loader))
    print("Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")
