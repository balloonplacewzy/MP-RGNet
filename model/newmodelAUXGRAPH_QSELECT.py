
from collections.abc import Mapping
import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from models.newmodelAUXGRAPH_AQ import Model as AQModel, observation_quality_loss


def resolve_config(configs):
    result = dict(vars(configs))
    integers = {"weather_bins": 8, "weather_dim": 16, "hidden_dim": 16,
                "message_hidden": 32, "gate_hidden": 32, "topk": 8,
                "state_hidden": int(configs.restore_hidden_dim)}
    floats = {"weather_temperature": 0.5, "prior_max": 2.0, "quality_temperature": 0.15,
              "window_fraction": 0.25, "receiver_fraction": 0.5, "window_threshold": 0.10,
              "point_threshold": 0.075, "source_threshold": 0.15, "margin": 0.025,
              "message_bound": 0.5, "state_init_bias": -2.3, "gate_init_bias": -1.0}
    for name, default in integers.items():
        key = "select_" + name
        value = float(getattr(configs, key, default))
        if not math.isfinite(value) or value < 1 or not value.is_integer():
            raise ValueError(f"{key} must be a positive integer")
        result[key] = int(value)
    for name, default in floats.items():
        key = "select_" + name
        value = float(getattr(configs, key, default))
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if name in ("weather_temperature", "quality_temperature", "message_bound") and value <= 0:
            raise ValueError(f"{key} must be positive")
        if name in ("window_fraction", "receiver_fraction") and not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0,1]")
        if name in ("prior_max", "window_threshold", "point_threshold", "source_threshold", "margin") and value < 0:
            raise ValueError(f"{key} must be nonnegative")
        result[key] = value
    for name in ("graph_enabled", "use_quality"):
        key = "select_" + name
        value = getattr(configs, key, True)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        result[key] = value
    result.update(graph_mode="off", model_name="newmodelAUXGRAPH_QSELECT",
                  select_receiver_count_rule="ceil(select_receiver_fraction * N)",
                  select_window_count_rule="ceil(select_window_fraction * L)")
    return result


class SelectiveStateGraph(nn.Module):

    def __init__(self, configs):
        super().__init__()
        cfg = resolve_config(configs)
        self.resolved_config = cfg
        self.num_sites = int(configs.target_dim)
        if self.num_sites < 1:
            raise ValueError("target_dim must be positive")
        for name in ("weather_bins", "weather_dim", "hidden_dim", "topk", "weather_temperature",
                     "quality_temperature", "prior_max", "receiver_fraction", "window_threshold",
                     "point_threshold", "source_threshold", "margin", "message_bound"):
            setattr(self, name, cfg["select_" + name])
        self.weather_projection = nn.Linear(int(configs.aux_per_site) * self.weather_bins, self.weather_dim)
        self.hidden_projection = nn.Linear(int(configs.restore_hidden_dim), self.hidden_dim)
        self.message_net = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + int(configs.aux_per_site) + 5, cfg["select_message_hidden"]),
            nn.GELU(), nn.Linear(cfg["select_message_hidden"], 1))
        nn.init.zeros_(self.message_net[-1].weight)
        nn.init.zeros_(self.message_net[-1].bias)
        self.gate_net = nn.Sequential(nn.Linear(self.hidden_dim + 4, cfg["select_gate_hidden"]),
                                      nn.GELU(), nn.Linear(cfg["select_gate_hidden"], 1))
        nn.init.constant_(self.gate_net[-1].bias, cfg["select_gate_init_bias"])
        self.prior_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("relation_prior", torch.zeros(self.num_sites, self.num_sites))

    @torch.no_grad()
    def set_relation_prior(self, prior, split_name="train"):
        """Store a caller-declared training prior; provenance needs caller audit."""
        if not isinstance(split_name, str) or not split_name.lower().startswith("train"):
            raise ValueError("The relation prior must be computed from training data only")
        prior = torch.as_tensor(prior, device=self.relation_prior.device, dtype=self.relation_prior.dtype)
        if prior.shape != self.relation_prior.shape or not torch.isfinite(prior).all():
            raise ValueError("Expected a finite [N,N] relation prior")
        if (prior.abs() > 1.000001).any():
            raise ValueError("Correlation entries must lie in [-1,1]")
        self.relation_prior.copy_(prior)
        self.relation_prior.fill_diagonal_(0)
        return self

    @staticmethod
    def gather(values, indices):
        batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
        return values.transpose(1, 2)[batch, indices].permute(0, 3, 1, 2, 4)

    @staticmethod
    def identity(state):
        B, L, N = state.shape
        zero = torch.zeros_like(state)
        disabled = torch.zeros_like(state, dtype=torch.bool)
        return {"p_graph": state, "delta": zero, "receiver_mask": disabled,
                "receiver_candidate": disabled, "source_mass": zero, "gate": zero,
                "weighted_source_error": zero, "attention": state.new_zeros(B, L, N, 0),
                "candidate_attention": state.new_zeros(B, L, N, 0),
                "source_mask": torch.empty(B, L, N, 0, device=state.device, dtype=torch.bool),
                "neighbor_idx": torch.empty(B, N, 0, device=state.device, dtype=torch.long)}

    def forward(self, features, use_quality=True):
        state, base = features["p_impute"], features["p_base"]
        hidden, aux, mask = features["hidden"], features["aux"], features["mask"]
        B, L, N = state.shape
        if N != self.num_sites:
            raise ValueError("Graph site count differs from the configuration")
        if N == 1:
            return self.identity(state)
        weather = aux.permute(0, 2, 3, 1).reshape(B * N, aux.shape[-1], L)
        weather = F.adaptive_avg_pool1d(weather, self.weather_bins).flatten(1)
        weather = F.normalize(self.weather_projection(weather), dim=-1, eps=1e-8).reshape(B, N, self.weather_dim)
        score = torch.matmul(weather, weather.transpose(1, 2)) / self.weather_temperature
        score = score + self.prior_max * torch.sigmoid(self.prior_logit) * self.relation_prior
        if use_quality:
            error = features["state_error"].detach()
            window_error = features["window_error"].detach()
            score = score - window_error[:, None, :] / self.quality_temperature
            largest = window_error.topk(max(1, math.ceil(self.receiver_fraction * N)), dim=-1).indices
            window_receiver = torch.zeros_like(window_error, dtype=torch.bool).scatter_(1, largest, True)
            window_receiver = window_receiver & (window_error > self.window_threshold)
            receiver_candidate = window_receiver[:, None, :] & (error > self.point_threshold)
        else:
            # Never inspect any quality tensor in this ablation.
            error = torch.zeros_like(state)
            receiver_candidate = torch.ones_like(state, dtype=torch.bool)
        off_diagonal = ~torch.eye(N, device=state.device, dtype=torch.bool)[None].expand(B, N, N)
        score = score.masked_fill(~off_diagonal, torch.finfo(score.dtype).min)
        values, indices = score.topk(min(self.topk, N - 1), dim=-1)
        valid_edges = off_diagonal.gather(-1, indices)
        A = torch.softmax(values, dim=-1) * valid_edges.to(values.dtype)
        A = A[:, None].expand(B, L, N, indices.shape[-1])
        source_error = self.gather(error[..., None], indices).squeeze(-1)
        source_mask = valid_edges[:, None].expand(B, L, N, indices.shape[-1])
        logits = A.clamp_min(torch.finfo(A.dtype).tiny).log()
        if use_quality:
            source_mask = source_mask & (source_error < self.source_threshold)
            source_mask = source_mask & (source_error + self.margin < error[..., None])
            logits = logits - source_error / self.quality_temperature
        source_mass = (A * source_mask.to(A.dtype)).sum(dim=-1)
        logits = logits.masked_fill(~source_mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1) * source_mask.to(logits.dtype)
        weighted_error = (attention * source_error).sum(dim=-1).detach()
        receiver_mask = receiver_candidate & source_mask.any(dim=-1)

        projected = self.hidden_projection(hidden)
        source_hidden = self.gather(projected, indices)
        target_hidden = projected.unsqueeze(3).expand_as(source_hidden)
        source_state = self.gather(state[..., None], indices)
        source_base = self.gather(base[..., None], indices)
        source_observed = self.gather(mask[..., None], indices)
        target_observed = mask[..., None, None].expand_as(source_observed)
        pair = torch.cat((target_hidden, source_hidden, self.gather(aux, indices) - aux.unsqueeze(3),
                          source_state - state[..., None, None], source_base - base[..., None, None],
                          source_state - source_base, source_observed, target_observed), dim=-1)
        message = self.message_bound * torch.tanh(self.message_net(pair).squeeze(-1))
        missing_fraction = (mask <= 0.5).to(state.dtype).mean(dim=1, keepdim=True).expand(-1, L, -1)
        gate_inputs = torch.cat((projected, missing_fraction[..., None], source_mass[..., None],
                                 error[..., None], weighted_error[..., None]), dim=-1)
        raw_gate = torch.sigmoid(self.gate_net(gate_inputs).squeeze(-1))
        gate = torch.where(receiver_mask, raw_gate, torch.zeros_like(raw_gate))
        correction = gate * (attention * message).sum(dim=-1)
        delta = torch.where(receiver_mask, correction, torch.zeros_like(correction))
        return {"p_graph": torch.where(receiver_mask, state + delta, state), "delta": delta,
                "receiver_mask": receiver_mask, "receiver_candidate": receiver_candidate,
                "source_mass": source_mass, "gate": gate, "weighted_source_error": weighted_error,
                "attention": attention, "candidate_attention": A, "source_mask": source_mask,
                "neighbor_idx": indices}


class Model(AQModel):

    def __init__(self, configs):
        resolved = resolve_config(configs)
        super().__init__(SimpleNamespace(**resolved))
        self.resolved_config = resolved
        self.window_fraction = resolved["select_window_fraction"]
        self.state_error_head = nn.Sequential(
            nn.Linear(int(configs.restore_hidden_dim) + 7, resolved["select_state_hidden"]),
            nn.GELU(), nn.Linear(resolved["select_state_hidden"], 1), nn.Softplus())
        nn.init.constant_(self.state_error_head[-2].bias, resolved["select_state_init_bias"])
        self.state_graph = SelectiveStateGraph(SimpleNamespace(**resolved))
        self.graph_enabled = resolved["select_graph_enabled"]
        self.use_quality = resolved["select_use_quality"]
        self._training_stage = "graph"
        self._freeze_for_stage()

    @staticmethod
    def error_features(features):
        obs, base = features["observation"], features["p_base"]
        scalar = torch.stack((obs, base, features["p_impute"], obs - base, (obs - base).abs(),
                              features["mask"], features["fusion_alpha"]), dim=-1)
        return torch.cat((features["hidden"], scalar), dim=-1).detach()

    def state_quality_outputs(self, features):
        error = self.state_error_head(self.error_features(features)).squeeze(-1)
        count = max(1, math.ceil(self.window_fraction * error.shape[1]))
        window_error = error.topk(count, dim=1).values.mean(dim=1)
        quality, window_quality = 1.0 / (1.0 + error), 1.0 / (1.0 + window_error)
        return {"state_error": error, "estimated_state_error": error, "state_quality": quality,
                "window_error": window_error, "window_quality": window_quality, "window_score": window_quality}

    def encode(self, x):
        observation, mask, aux = self.splitter(x)
        state, base, alpha, hidden = self.restore_head(observation, mask, aux)
        features = {"p_base": base, "p_impute": state, "hidden": hidden, "observation": observation,
                    "mask": mask, "aux": aux, "fusion_alpha": alpha}
        obs_error = self.obs_error_head(self.error_features(features)).squeeze(-1)
        features.update(estimated_obs_error=obs_error, obs_quality=mask / (1.0 + obs_error),
                        obs_quality_valid=mask > 0.5)
        features.update(self.state_quality_outputs(features))
        return features

    def forward_from_features(self, features):
        values = dict(features)
        if self._training_stage == "state_quality":
            values.update(self.state_quality_outputs(values))
        graph = (self.state_graph(values, use_quality=self.use_quality) if self.graph_enabled
                 else self.state_graph.identity(values["p_impute"]))
        keys = ("p_base", "p_impute", "fusion_alpha", "estimated_obs_error", "obs_quality", "obs_quality_valid",
                "state_error", "estimated_state_error", "state_quality", "window_error", "window_quality", "window_score")
        output = {key: values[key] for key in keys}
        output.update(graph)
        output["pred"] = self.forecast_head(graph["p_graph"])
        return output

    def forward(self, x, batch=None):
        del batch
        return self.forward_from_features(self.encode(x))

    def load_aq_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("Expected the AQ checkpoint's direct state_dict")
        current = self.state_dict()
        expected = {name for name in current if not name.startswith(("state_error_head.", "state_graph."))}
        if set(state) != expected:
            raise RuntimeError(f"AQ state mismatch: missing={sorted(expected - set(state))}, "
                               f"unexpected={sorted(set(state) - expected)}")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != current[name].shape:
                raise RuntimeError(f"AQ state shape/type mismatch: {name}")
        current.update(state)
        self.load_state_dict(current, strict=True)
        return {"loaded_aq_keys": sorted(expected), "new_state_keys": sorted(set(current) - expected)}

    def set_relation_prior(self, prior, split_name="train"):
        self.state_graph.set_relation_prior(prior, split_name=split_name)
        return self

    def _freeze_for_stage(self):
        prefix = "state_error_head." if self._training_stage == "state_quality" else "state_graph."
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
            parameter.grad = None
        self.train(self.training)
        return self

    def train_state_quality_only(self):
        self._training_stage = "state_quality"
        self.graph_enabled = False
        return self._freeze_for_stage()

    def train_graph_only(self):
        self._training_stage = "graph"
        self.graph_enabled = True
        return self._freeze_for_stage()

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "_training_stage"):
            self.splitter.eval()
            self.restore_head.eval()
            self.forecast_head.eval()
            self.obs_error_head.eval()
            self.state_error_head.train(mode and self._training_stage == "state_quality")
            self.state_graph.train(mode and self._training_stage == "graph")
        return self


def state_quality_loss(output, batch):

    estimate = output["state_error"]
    state = output["p_impute"].detach()
    reference = batch["pv_clean"].detach().to(estimate)
    if reference.shape != estimate.shape or state.shape != estimate.shape:
        raise ValueError("Reference, reconstruction and estimated errors must have equal shapes")
    valid = torch.isfinite(reference) & torch.isfinite(state)
    if not valid.any():
        return estimate.reshape(-1)[:0].sum()
    target = (state[valid] - reference[valid]).abs().detach()
    return (estimate[valid] - target).square().mean()
