

from collections.abc import Mapping

import torch
import torch.nn as nn

from models.newmodelAUXGRAPH import (
    InputSplitter,
    MeteoPowerTimeVariableFACTBlock,
    AntiNoiseTemporalForecastHead,
    WindowMeteoReliabilityGraphSmoother,
)


class WeatherPowerStateReconstructionHead(nn.Module):


    def __init__(self, configs):
        super().__init__()
        self.num_sites = int(configs.target_dim)
        self.aux_per_site = int(configs.aux_per_site)
        self.hidden_dim = int(configs.restore_hidden_dim)
        self.swr_index = int(configs.swr_index)
        self.fact_layers = int(configs.restore_fact_layers)
        self.power_encoder = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.swr_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.site_embedding = nn.Parameter(
            torch.randn(self.num_sites, self.hidden_dim) * 0.02
        )
        self.fact_blocks = nn.ModuleList([
            MeteoPowerTimeVariableFACTBlock(configs)
            for _ in range(self.fact_layers)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.p_base_head = nn.Linear(self.hidden_dim, 1)
        self.alpha_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, p_obs: torch.Tensor, obs_mask: torch.Tensor, aux: torch.Tensor):
        # p_obs, obs_mask: [B, L, N]; aux: [B, L, N, K]
        B, L, N = p_obs.shape
        site_emb = self.site_embedding.view(1, 1, N, self.hidden_dim)
        power_input = torch.stack([p_obs, obs_mask], dim=-1)
        power_token = self.power_encoder(power_input) + site_emb
        swr = aux[..., self.swr_index:self.swr_index + 1]
        swr_token = self.swr_encoder(swr) + site_emb

        context = power_token
        for block in self.fact_blocks:
            context = block(context, swr_token)
        hidden = self.fusion(torch.cat([power_token, context], dim=-1))
        p_base = self.p_base_head(hidden).squeeze(-1)
        innovation = torch.abs(p_obs - p_base).unsqueeze(-1)
        alpha_input = torch.cat(
            [hidden, innovation, obs_mask.unsqueeze(-1)], dim=-1
        )
        alpha = torch.sigmoid(self.alpha_head(alpha_input).squeeze(-1))
        p_restore = (
            obs_mask * (alpha * p_obs + (1.0 - alpha) * p_base)
            + (1.0 - obs_mask) * p_base
        )
        return p_restore, p_base, alpha, hidden


class Model(nn.Module):


    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.num_sites = int(configs.target_dim)
        self.target_dim = int(configs.target_dim)
        self.enc_in = int(configs.enc_in)
        self.aux_per_site = int(configs.aux_per_site)
        self.graph_mode = str(getattr(configs, "graph_mode", "off"))
        if self.graph_mode not in ("off", "legacy_alpha"):
            raise ValueError("graph_mode must be 'off' or 'legacy_alpha'")
        self._quality_only_training = False

        self.splitter = InputSplitter(self.num_sites, self.aux_per_site)
        self.restore_head = WeatherPowerStateReconstructionHead(configs)
        if self.graph_mode == "legacy_alpha":
            self.meteo_graph_smoother = WindowMeteoReliabilityGraphSmoother(configs)
        else:
            self.meteo_graph_smoother = None

        self.forecast_head = AntiNoiseTemporalForecastHead(configs)
        hidden_dim = int(configs.restore_hidden_dim)
        self.obs_error_head = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, batch=None):

        del batch
        p_obs, obs_mask, aux = self.splitter(x)
        p_restore, p_base, alpha, hidden = self.restore_head(p_obs, obs_mask, aux)

        error_features = torch.stack(
            [p_obs, p_base, p_restore, p_obs - p_base,
             torch.abs(p_obs - p_base), obs_mask, alpha],
            dim=-1,
        )
        error_features = torch.cat([hidden, error_features], dim=-1).detach()
        estimated_obs_error = self.obs_error_head(error_features).squeeze(-1)
        obs_quality_valid = obs_mask > 0.5
        obs_quality = obs_mask / (1.0 + estimated_obs_error)

        if self.graph_mode == "legacy_alpha":

            p_graph, _ = self.meteo_graph_smoother(
                p_restore=p_restore,
                reliability=alpha,
                obs_mask=obs_mask,
                aux=aux,
            )
        else:
            p_graph = p_restore
        y_hat = self.forecast_head(p_graph)
        return {
            "pred": y_hat,
            "p_impute": p_restore,
            "p_graph": p_graph,
            "p_base": p_base,
            "fusion_alpha": alpha,
            "estimated_obs_error": estimated_obs_error,
            "obs_quality": obs_quality,
            "obs_quality_valid": obs_quality_valid,
        }

    def load_original_state_dict(self, state):

        if not isinstance(state, Mapping):
            raise TypeError("state must be the original model's direct state_dict")
        current = self.state_dict()
        new_keys = {key for key in current if key.startswith("obs_error_head.")}
        expected_keys = set(current) - new_keys
        graph_prefix = "meteo_graph_smoother."
        old_graph_keys = {
            graph_prefix + "gate_logit", graph_prefix + "aux_indices"
        }
        incoming_graph_keys = {key for key in state if key.startswith(graph_prefix)}
        if self.graph_mode == "off" and incoming_graph_keys:
            if incoming_graph_keys != old_graph_keys:
                raise RuntimeError(
                    "Original graph entries are incomplete or unknown: "
                    f"{sorted(incoming_graph_keys)}"
                )
            if state[graph_prefix + "gate_logit"].shape != torch.Size([]):
                raise RuntimeError("Original graph gate_logit must be scalar")
            if state[graph_prefix + "aux_indices"].ndim != 1:
                raise RuntimeError("Original graph aux_indices must be one-dimensional")

        mapped = {}
        renamed = {}
        ignored = []
        source_prefix = "restore_head.reliability_head."
        target_prefix = "restore_head.alpha_head."
        for key, value in state.items():
            if self.graph_mode == "off" and key in old_graph_keys:
                ignored.append(key)
                continue
            target_key = target_prefix + key[len(source_prefix):] if key.startswith(
                source_prefix
            ) else key
            if target_key in mapped:
                raise RuntimeError(f"Duplicate migrated parameter: {target_key}")
            if key != target_key:
                renamed[key] = target_key
            mapped[target_key] = value

        missing = expected_keys - set(mapped)
        unexpected = set(mapped) - expected_keys
        if missing or unexpected:
            raise RuntimeError(
                f"Original checkpoint mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        for key, value in mapped.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Original state entry {key} must be a tensor")
            if value.shape != current[key].shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}: {tuple(value.shape)} "
                    f"vs expected {tuple(current[key].shape)}"
                )
        complete_state = dict(current)
        complete_state.update(mapped)
        self.load_state_dict(complete_state, strict=True)
        return {
            "renamed_keys": renamed,
            "ignored_graph_keys": sorted(ignored),
            "new_quality_keys": sorted(new_keys),
            "loaded_backbone_keys": sorted(mapped),
            "graph_mode": self.graph_mode,
        }

    def freeze_backbone_for_quality_training(self):

        self._quality_only_training = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.obs_error_head.parameters():
            parameter.requires_grad_(True)
        self.train(True)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if self._quality_only_training:
            self.splitter.eval()
            self.restore_head.eval()
            self.forecast_head.eval()
            if self.meteo_graph_smoother is not None:
                self.meteo_graph_smoother.eval()
            self.obs_error_head.train(mode)
        return self


def observation_quality_loss(output, batch):

    estimated_error = output["estimated_obs_error"]
    obs = batch["pv_obs"].detach().to(estimated_error)
    clean = batch["pv_clean"].detach().to(estimated_error)
    mask = batch["obs_mask"].detach().to(device=estimated_error.device) > 0.5
    quality_valid = output["obs_quality_valid"].detach().to(
        device=estimated_error.device, dtype=torch.bool
    )
    for name, value in (
        ("pv_obs", obs), ("pv_clean", clean),
        ("obs_mask", mask), ("obs_quality_valid", quality_valid),
    ):
        if value.shape != estimated_error.shape:
            raise ValueError(f"{name} must have shape {tuple(estimated_error.shape)}")
    valid = mask & quality_valid & torch.isfinite(obs) & torch.isfinite(clean)
    target = torch.where(valid, torch.abs(obs - clean), torch.zeros_like(obs)).detach()
    estimate = torch.where(valid, estimated_error, torch.zeros_like(estimated_error))
    squared_error = (estimate - target).square()
    if "day_mask" in batch:
        day = batch["day_mask"].detach().to(device=estimated_error.device)
        if day.shape != estimated_error.shape:
            raise ValueError(f"day_mask must have shape {tuple(estimated_error.shape)}")
        known_day = torch.isfinite(day)
        groups = [valid & known_day & (day > 0.5), valid & known_day & (day <= 0.5)]
    else:
        groups = [valid]
    group_losses = [squared_error[group].mean() for group in groups if group.any()]
    if not group_losses:
        return estimated_error.reshape(-1)[:0].sum()
    return torch.stack(group_losses).mean()
