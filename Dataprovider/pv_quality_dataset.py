import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
from importlib.metadata import version
import platform
from typing import Dict, Optional, Tuple, Any
from sklearn.impute import KNNImputer


AUX_DELAY_STEPS = 4                                                                     
DATA_PROTOCOL_VERSION = "pv-quality-v2"
DATA_OPTION_FIELDS = ("aux_delay_steps", "missing_pattern", "block_len", "data_seed", "append_obs_mask")













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



    knn_n_neighbors: int = 5
    knn_weights: str = "distance"






    aux_layout: str = "feature_major"

    aux_delay_steps: Optional[int] = None
    missing_pattern: Optional[str] = None
    block_len: Optional[int] = None
    data_seed: Optional[int] = None
    append_obs_mask: Optional[bool] = None


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _finite_number(value, name, minimum=0.0, upper=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < minimum or (upper is not None and value >= upper)):
        raise ValueError(f"Invalid {name}={value!r}; expected finite value >= {minimum}"
                         + (f" and < {upper}" if upper is not None else ""))


def validate_missing_parameters(missing_rate, missing_pattern, block_len):
    _finite_number(missing_rate, "missing_rate", upper=1.0)
    if missing_pattern not in ("none", "random_point", "block", "station"):
        raise ValueError(f"Unknown missing_pattern={missing_pattern!r}")
    _integer(block_len, "block_len", minimum=1)


def resolve_scenario(cfg, scenario_override=None):

    scenario = get_scenario(cfg.scenario_name)
    scenario.setdefault("block_len", 8)
    for key in ("aux_delay_steps", "missing_pattern", "block_len"):
        value = getattr(cfg, key)
        if value is not None:
            scenario[key] = value
    if scenario_override is not None:
        if not isinstance(scenario_override, dict):
            raise ValueError("scenario_override must be a dict")
        unknown = set(scenario_override) - set(scenario)
        if unknown:
            raise ValueError(f"Unknown scenario_override fields: {sorted(unknown)}")
        scenario.update(scenario_override)
    _integer(scenario["aux_delay_steps"], "aux_delay_steps")
    _finite_number(scenario["noise_std"], "noise_std")
    validate_missing_parameters(scenario["missing_rate"], scenario["missing_pattern"], scenario["block_len"])
    return {key: (int(value) if key in ("aux_delay_steps", "block_len") else
                  float(value) if key in ("noise_std", "missing_rate") else value)
            for key, value in scenario.items()}


def degradation_generators(data_seed):

    _integer(data_seed, "data_seed")
    return tuple(np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(int(data_seed), spawn_key=(stream_id,))))
        for stream_id in (0, 1))


def describe_missingness(obs_mask, train_end, val_end, block_mode):

    result = {}
    for name, start, end in (("train", 0, train_end), ("val", train_end, val_end),
                             ("test", val_end, len(obs_mask))):
        missing = obs_mask[start:end] < 0.5
        stats = {"start": start, "end": end, "missing_count": int(missing.sum()),
                 "actual_missing_rate": float(missing.mean()) if missing.size else None}
        if block_mode:
            lengths = []
            for site in range(missing.shape[1]):
                edges = np.diff(np.pad(missing[:, site].astype(np.int8), (1, 1)))
                lengths.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
            stats.update(run_count=len(lengths), mean_run_length=float(np.mean(lengths)) if lengths else 0.0,
                         max_run_length=max(lengths, default=0))
        result[name] = stats
    return result


def add_data_arguments(parser):
    parser.add_argument("--aux_delay_steps", type=int, default=None)
    parser.add_argument("--missing_pattern", choices=["none", "random_point", "block", "station"], default=None)
    parser.add_argument("--block_len", type=int, default=None,
                        help="Length of each placed block, not final merged runs.")
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--append_obs_mask", action=argparse.BooleanOptionalAction, default=None,
                        help="Only controls mask concatenation into x; use --no-append_obs_mask to disable.")
    parser.add_argument("--aux_layout", choices=["feature_major", "unit_major"], default=None)


def validate_model_data_interface(args, meta):

    if args.model_name == "newmodelAUXGRAPH" and not meta["append_obs_mask"]:
        raise ValueError("newmodelAUXGRAPH requires append_obs_mask=True: input is [PV, obs_mask, aux].")
    if args.model_type == "baseline" and meta["input_dim"] != args.enc_in:
        raise ValueError(f"Model enc_in={args.enc_in}, but dataset input_dim={meta['input_dim']}. "
                         "Mask concatenation does not adapt the model; check the input configuration.")


def prepare_run_outputs(args, meta):

    dataset_name = Path(args.data).stem
    checkpoint_dir = Path(args.checkpoint_root) / DATA_PROTOCOL_VERSION / dataset_name / args.scenario_name
    result_dir = Path(args.result_root) / DATA_PROTOCOL_VERSION / dataset_name / args.scenario_name
    stage = f"_stage-{args.stage}" if hasattr(args, "stage") else ""
    run_identifier = (f"{args.model_name}_type-{args.model_type}{stage}_scenario-{args.scenario_name}"
                      f"_imp-{args.impute_method}_len{args.seq_len}-{args.pred_len}_seed{args.seed}"
                      f"_data-{meta['data_config_id']}")
    paths = {
        "checkpoint_path": checkpoint_dir / f"{run_identifier}.pth",
        "first_step_csv_path": result_dir / f"{run_identifier}_first_step.csv",
        "overlap_avg_csv_path": result_dir / f"{run_identifier}_overlap_avg.csv",
        "config_path": result_dir / f"{run_identifier}_config.json",
        "metrics_csv_path": result_dir / "metrics_summary.csv",
    }
    for key, path in paths.items():
        if key != "metrics_csv_path" and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing run artifact: {path}")
    args.run_identifier = run_identifier
    record = {"run_identifier": run_identifier, "data_protocol_version": DATA_PROTOCOL_VERSION,
              "data_config_id": meta["data_config_id"], "effective_data_config": meta["effective_data_config"],
              "requested_data_config": meta["requested_data_config"], "arguments": vars(args),
              "missing_statistics": meta["missing_statistics"],
              "missing_statistics_boundary": meta["missing_statistics_boundary"],
              "environment": {"python": platform.python_version(),
                              **{name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")}}}
    serialized = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    with paths["config_path"].open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return paths


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







    _integer(delay_steps, "delay_steps")
    if delay_steps == 0 or len(data) == 0:
        return data.copy()

    delayed = np.empty_like(data)
    delayed[:delay_steps] = data[0:1]
    if delay_steps < len(data):
        delayed[delay_steps:] = data[:-delay_steps]
    return delayed.astype(np.float32)


def add_daytime_gaussian_noise(
    pv_obs_scaled: np.ndarray,
    pv_obs_raw: np.ndarray,
    noise_std: float,
    day_threshold: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:

















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






    T, N = shape
    _integer(T, "T", minimum=1)
    _integer(N, "N", minimum=1)
    validate_missing_parameters(missing_rate, missing_pattern, block_len)
    if missing_rate == 0 or missing_pattern == "none":
        return np.ones((T, N), dtype=np.float32)

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


    imputer.fit(safe_data[:train_end])


    imputed_data = imputer.transform(safe_data).astype(np.float32)


    imputed_data = np.nan_to_num(imputed_data, nan=0.0).astype(np.float32)

    return imputed_data, imputer


def fill_missing_values(
    data: np.ndarray,
    method: str,
    train_initial_values: np.ndarray,
) -> np.ndarray:










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
    _integer(cfg.seed if cfg.data_seed is None else cfg.data_seed, "effective data_seed")
    if cfg.append_obs_mask is not None and not isinstance(cfg.append_obs_mask, (bool, np.bool_)):
        raise ValueError("append_obs_mask must be None, True or False")
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
            "x": torch.from_numpy(x),                                                                                                                
            "y": torch.from_numpy(y),                                           
            "pv_obs": torch.from_numpy(pv_obs),                                
            "obs_mask": torch.from_numpy(obs_mask),                            
            "pv_clean": torch.from_numpy(pv_clean_hist),                                                     
            "day_mask": torch.from_numpy(day_mask),                            
            "quality": torch.from_numpy(quality),                                                         
            "y_raw": torch.from_numpy(y_raw),                                   
            "index": torch.tensor(s, dtype=torch.long),
        }

        return batch


def build_pv_quality_dataloaders(
    cfg: PVDataConfig,
    scenario_override: Optional[Dict[str, Any]] = None,
):


























    validate_config(cfg)

    scenario = resolve_scenario(cfg, scenario_override)
    data_seed = int(cfg.seed if cfg.data_seed is None else cfg.data_seed)
    noise_rng, missing_rng = degradation_generators(data_seed)

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


    pv_scaler.fit(pv_raw[:train_end])
    aux_scaler.fit(aux_raw[:train_end])

    pv_scaled = pv_scaler.transform(pv_raw).astype(np.float32)
    aux_scaled = aux_scaler.transform(aux_raw).astype(np.float32)






    pv_obs_scaled = pv_scaled.copy().astype(np.float32)
    pv_obs_raw = pv_raw.copy().astype(np.float32)





    noise_std = float(scenario["noise_std"])
    pv_obs_scaled, day_mask = add_daytime_gaussian_noise(
        pv_obs_scaled=pv_obs_scaled,
        pv_obs_raw=pv_obs_raw,
        noise_std=noise_std,
        day_threshold=cfg.day_threshold,
        rng=noise_rng,
    )





    obs_mask = generate_missing_mask(
        shape=pv_obs_scaled.shape,
        missing_rate=float(scenario["missing_rate"]),
        missing_pattern=str(scenario["missing_pattern"]),
        rng=missing_rng,
        block_len=scenario["block_len"],
    )
    pv_obs_with_nan = apply_missing(pv_obs_scaled, obs_mask)








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





    aux_delay_steps = int(scenario["aux_delay_steps"])
    aux_model_input = time_delay(aux_scaled, aux_delay_steps)






    append_obs_mask = (cfg.model_type == "ours") if cfg.append_obs_mask is None else bool(cfg.append_obs_mask)
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


    effective = {
        "data_protocol_version": DATA_PROTOCOL_VERSION,
        "csv_sha256": hashlib.sha256(Path(cfg.csv_path).read_bytes()).hexdigest(),
        "data_seed": data_seed, **scenario,
        "append_obs_mask": append_obs_mask, "impute_method": cfg.impute_method,
        "knn_n_neighbors": int(cfg.knn_n_neighbors), "knn_weights": cfg.knn_weights,
        "num_sites": int(cfg.num_sites), "aux_layout": cfg.aux_layout,
        "scale": bool(cfg.scale), "day_threshold": float(cfg.day_threshold),
        "train_ratio": float(cfg.train_ratio), "val_ratio": float(cfg.val_ratio),
        "total_len": total_len, "train_end": train_end, "val_end": val_end,
    }
    config_id = hashlib.sha256(json.dumps(effective, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8")).hexdigest()[:12]
    meta.update(data_seed=data_seed, data_protocol_version=DATA_PROTOCOL_VERSION,
                data_config_id=config_id, effective_data_config=effective,
                requested_data_config=asdict(cfg),
                missing_statistics=describe_missingness(obs_mask, train_end, val_end,
                                                       scenario["missing_pattern"] == "block"),
                missing_statistics_boundary="Per-site runs clipped to each [start, end) split; "
                                            "a run crossing a boundary is counted in both intervals.")

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
