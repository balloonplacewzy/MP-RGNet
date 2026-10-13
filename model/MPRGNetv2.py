import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F


# Input and meteorological feature extraction

class InputSplitter(nn.Module):

    def __init__(self, num_sites: int, aux_per_site: int):
        super().__init__()
        self.num_sites = int(num_sites)
        self.aux_per_site = int(aux_per_site)

    def forward(self, x: torch.Tensor):
        n = self.num_sites
        k = self.aux_per_site
        if x.ndim != 3 or x.shape[-1] != n * (k + 2):
            raise ValueError('Expected x=[PV, observed mask, site-major weather], shape [B,L,N*(K+2)]')
        p_obs = x[:, :, :n]
        obs_mask = x[:, :, n:2 * n]
        aux = x[:, :, 2 * n:]
        aux = aux.reshape(x.shape[0], x.shape[1], n, k)
        return (p_obs, obs_mask, aux)

class MeteoPowerTimeVariableFACTBlock(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.hidden_dim = int(configs.restore_hidden_dim)
        self.kernel_time = int(configs.restore_kernel_time)
        self.kernel_var = int(configs.restore_kernel_var)
        self.dilations = tuple((int(d) for d in configs.restore_dilations))
        self.dropout = float(configs.restore_dropout)
        self.branches = nn.ModuleList()
        for d in self.dilations:
            pad_t = d * (self.kernel_time - 1) // 2
            pad_v = d * (self.kernel_var - 1) // 2
            self.branches.append(nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=(self.kernel_time, self.kernel_var), padding=(pad_t, pad_v), dilation=(d, d), groups=self.hidden_dim, bias=True), nn.GELU()))
        self.fuse = nn.Sequential(nn.Conv2d(in_channels=self.hidden_dim * len(self.branches), out_channels=self.hidden_dim, kernel_size=1, bias=True), nn.GELU(), nn.Dropout(self.dropout))
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, power_token: torch.Tensor, swr_token: torch.Tensor):
        B, L, N, D = power_token.shape
        z = torch.stack([power_token, swr_token], dim=3).reshape(B, L, 2 * N, D)
        x = z.permute(0, 3, 1, 2).contiguous()
        h = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.norm(h + z)
        h = h.reshape(B, L, N, 2, D)
        return h[:, :, :, 0, :]


# State reconstruction and fusion alpha

class WeatherPowerStateReconstructionHead(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.num_sites = int(configs.target_dim)
        self.aux_per_site = int(configs.aux_per_site)
        self.hidden_dim = int(configs.restore_hidden_dim)
        self.swr_index = int(configs.swr_index)
        self.fact_layers = int(configs.restore_fact_layers)
        self.power_encoder = nn.Sequential(nn.Linear(2, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.swr_encoder = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.site_embedding = nn.Parameter(torch.randn(self.num_sites, self.hidden_dim) * 0.02)
        self.fact_blocks = nn.ModuleList([MeteoPowerTimeVariableFACTBlock(configs) for _ in range(self.fact_layers)])
        self.fusion = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.GELU(), nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU())
        self.p_base_head = nn.Linear(self.hidden_dim, 1)
        self.alpha_head = nn.Sequential(nn.Linear(self.hidden_dim + 2, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, 1))

    def forward(self, p_obs: torch.Tensor, obs_mask: torch.Tensor, aux: torch.Tensor):
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
        alpha_input = torch.cat([hidden, innovation, obs_mask.unsqueeze(-1)], dim=-1)
        alpha = torch.sigmoid(self.alpha_head(alpha_input).squeeze(-1))
        p_restore = obs_mask * (alpha * p_obs + (1.0 - alpha) * p_base) + (1.0 - obs_mask) * p_base
        return (p_restore, p_base, alpha, hidden)


# Temporal forecast head

class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return (res, moving_mean)

class AntiNoiseTemporalForecastHead(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.hidden_size = int(configs.hidden_size)
        self.decompsition = series_decomp(25)
        self.projection = nn.Sequential(nn.Linear(self.seq_len, self.seq_len), nn.GELU(), nn.Linear(self.seq_len, self.pred_len))
        self.linear_seasonal = nn.Sequential(nn.Linear(self.seq_len, self.hidden_size), nn.LeakyReLU(), nn.Linear(self.hidden_size, self.pred_len))
        self.linear_trend = nn.Sequential(nn.Linear(self.seq_len, self.hidden_size), nn.LeakyReLU(), nn.Linear(self.hidden_size, self.pred_len))

    def forward(self, p_restore: torch.Tensor):
        seasonal, trend = self.decompsition(p_restore)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out = seasonal + trend
        return out


# Base state graph

def resolve_config(configs):
    result = dict(vars(configs))
    integers = {'weather_bins': 8, 'weather_dim': 16, 'hidden_dim': 16, 'message_hidden': 32, 'gate_hidden': 32, 'topk': 8, 'state_hidden': int(configs.restore_hidden_dim)}
    floats = {'weather_temperature': 0.5, 'prior_max': 2.0, 'quality_temperature': 0.15, 'window_fraction': 0.25, 'receiver_fraction': 0.5, 'window_threshold': 0.1, 'point_threshold': 0.075, 'source_threshold': 0.15, 'margin': 0.025, 'message_bound': 0.5, 'state_init_bias': -2.3, 'gate_init_bias': -1.0}
    for name, default in integers.items():
        key = 'select_' + name
        value = float(getattr(configs, key, default))
        if not math.isfinite(value) or value < 1 or (not value.is_integer()):
            raise ValueError(f'{key} must be a positive integer')
        result[key] = int(value)
    for name, default in floats.items():
        key = 'select_' + name
        value = float(getattr(configs, key, default))
        if not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if name in ('weather_temperature', 'quality_temperature', 'message_bound') and value <= 0:
            raise ValueError(f'{key} must be positive')
        if name in ('window_fraction', 'receiver_fraction') and (not 0 < value <= 1):
            raise ValueError(f'{key} must be in (0,1]')
        if name in ('prior_max', 'window_threshold', 'point_threshold', 'source_threshold', 'margin') and value < 0:
            raise ValueError(f'{key} must be nonnegative')
        result[key] = value
    for name in ('graph_enabled', 'use_quality'):
        key = 'select_' + name
        value = getattr(configs, key, True)
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be a boolean')
        result[key] = value
    result.update(graph_mode='off', model_name='newmodelAUXGRAPH_QSELECT', select_receiver_count_rule='ceil(select_receiver_fraction * N)', select_window_count_rule='ceil(select_window_fraction * L)')
    return result

class SelectiveStateGraph(nn.Module):

    def __init__(self, configs):
        super().__init__()
        cfg = resolve_config(configs)
        self.resolved_config = cfg
        self.num_sites = int(configs.target_dim)
        if self.num_sites < 1:
            raise ValueError('target_dim must be positive')
        for name in ('weather_bins', 'weather_dim', 'hidden_dim', 'topk', 'weather_temperature', 'quality_temperature', 'prior_max', 'receiver_fraction', 'window_threshold', 'point_threshold', 'source_threshold', 'margin', 'message_bound'):
            setattr(self, name, cfg['select_' + name])
        self.weather_projection = nn.Linear(int(configs.aux_per_site) * self.weather_bins, self.weather_dim)
        self.hidden_projection = nn.Linear(int(configs.restore_hidden_dim), self.hidden_dim)
        self.message_net = nn.Sequential(nn.Linear(2 * self.hidden_dim + int(configs.aux_per_site) + 5, cfg['select_message_hidden']), nn.GELU(), nn.Linear(cfg['select_message_hidden'], 1))
        nn.init.zeros_(self.message_net[-1].weight)
        nn.init.zeros_(self.message_net[-1].bias)
        self.gate_net = nn.Sequential(nn.Linear(self.hidden_dim + 4, cfg['select_gate_hidden']), nn.GELU(), nn.Linear(cfg['select_gate_hidden'], 1))
        nn.init.constant_(self.gate_net[-1].bias, cfg['select_gate_init_bias'])
        self.prior_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer('relation_prior', torch.zeros(self.num_sites, self.num_sites))

    @torch.no_grad()
    def set_relation_prior(self, prior, split_name='train'):
        if not isinstance(split_name, str) or not split_name.lower().startswith('train'):
            raise ValueError('The relation prior must be computed from training data only')
        prior = torch.as_tensor(prior, device=self.relation_prior.device, dtype=self.relation_prior.dtype)
        if prior.shape != self.relation_prior.shape or not torch.isfinite(prior).all():
            raise ValueError('Expected a finite [N,N] relation prior')
        if (prior.abs() > 1.000001).any():
            raise ValueError('Correlation entries must lie in [-1,1]')
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
        return {'p_graph': state, 'delta': zero, 'receiver_mask': disabled, 'receiver_candidate': disabled, 'source_mass': zero, 'gate': zero, 'weighted_source_error': zero, 'attention': state.new_zeros(B, L, N, 0), 'candidate_attention': state.new_zeros(B, L, N, 0), 'source_mask': torch.empty(B, L, N, 0, device=state.device, dtype=torch.bool), 'neighbor_idx': torch.empty(B, N, 0, device=state.device, dtype=torch.long)}

    def forward(self, features, use_quality=True):
        state, base = (features['p_impute'], features['p_base'])
        hidden, aux, mask = (features['hidden'], features['aux'], features['mask'])
        B, L, N = state.shape
        if N != self.num_sites:
            raise ValueError('Graph site count differs from the configuration')
        if N == 1:
            return self.identity(state)
        weather = aux.permute(0, 2, 3, 1).reshape(B * N, aux.shape[-1], L)
        weather = F.adaptive_avg_pool1d(weather, self.weather_bins).flatten(1)
        weather = F.normalize(self.weather_projection(weather), dim=-1, eps=1e-08).reshape(B, N, self.weather_dim)
        score = torch.matmul(weather, weather.transpose(1, 2)) / self.weather_temperature
        score = score + self.prior_max * torch.sigmoid(self.prior_logit) * self.relation_prior
        if use_quality:
            error = features['state_error'].detach()
            window_error = features['window_error'].detach()
            score = score - window_error[:, None, :] / self.quality_temperature
            largest = window_error.topk(max(1, math.ceil(self.receiver_fraction * N)), dim=-1).indices
            window_receiver = torch.zeros_like(window_error, dtype=torch.bool).scatter_(1, largest, True)
            window_receiver = window_receiver & (window_error > self.window_threshold)
            receiver_candidate = window_receiver[:, None, :] & (error > self.point_threshold)
        else:
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
        pair = torch.cat((target_hidden, source_hidden, self.gather(aux, indices) - aux.unsqueeze(3), source_state - state[..., None, None], source_base - base[..., None, None], source_state - source_base, source_observed, target_observed), dim=-1)
        message = self.message_bound * torch.tanh(self.message_net(pair).squeeze(-1))
        missing_fraction = (mask <= 0.5).to(state.dtype).mean(dim=1, keepdim=True).expand(-1, L, -1)
        gate_inputs = torch.cat((projected, missing_fraction[..., None], source_mass[..., None], error[..., None], weighted_error[..., None]), dim=-1)
        raw_gate = torch.sigmoid(self.gate_net(gate_inputs).squeeze(-1))
        gate = torch.where(receiver_mask, raw_gate, torch.zeros_like(raw_gate))
        correction = gate * (attention * message).sum(dim=-1)
        delta = torch.where(receiver_mask, correction, torch.zeros_like(correction))
        return {'p_graph': torch.where(receiver_mask, state + delta, state), 'delta': delta, 'receiver_mask': receiver_mask, 'receiver_candidate': receiver_candidate, 'source_mass': source_mass, 'gate': gate, 'weighted_source_error': weighted_error, 'attention': attention, 'candidate_attention': A, 'source_mask': source_mask, 'neighbor_idx': indices}


# Quality-guided graph propagation

class QualityStateGraph(SelectiveStateGraph):

    def __init__(self, configs):
        super().__init__(configs)
        self.quality_strength = float(getattr(configs, 'jointq_strength', 0.5))
        self.quality_scope = str(getattr(configs, 'jointq_scope', 'longgap'))
        self.gap_steps = int(getattr(configs, 'jointq_gap_steps', 24))
        if not math.isfinite(self.quality_strength) or self.quality_strength < 0:
            raise ValueError('jointq_strength must be finite and nonnegative')
        if self.quality_scope not in ('all', 'longgap') or self.gap_steps < 1:
            raise ValueError('Invalid quality scope/gap length')

    @staticmethod
    def longest_gap(mask):
        steps = torch.arange(1, mask.shape[1] + 1, device=mask.device).view(1, -1, 1)
        last_observed = torch.where(mask > 0.5, steps, 0).cummax(dim=1).values
        return (steps - last_observed).amax(dim=1)

    def forward(self, features, mode='soft'):
        if mode == 'hard':
            return super().forward(features, use_quality=True)
        if mode == 'off' or self.quality_strength == 0:
            return super().forward(features, use_quality=False)
        if mode != 'soft':
            raise ValueError(mode)
        state, base = (features['p_impute'], features['p_base'])
        hidden, aux, mask = (features['hidden'], features['aux'], features['mask'])
        B, L, N = state.shape
        if N == 1:
            return self.identity(state)
        error, window_error = (features['state_error'].detach(), features['window_error'].detach())
        largest = window_error.topk(max(1, math.ceil(self.receiver_fraction * N)), dim=-1).indices
        selected = torch.zeros_like(window_error, dtype=torch.bool).scatter_(1, largest, True)
        selected &= window_error > self.window_threshold
        if self.quality_scope == 'longgap':
            selected &= self.longest_gap(mask) >= self.gap_steps
        receiver = selected[:, None, :] & (error > self.point_threshold)
        weather = aux.permute(0, 2, 3, 1).reshape(B * N, aux.shape[-1], L)
        weather = F.adaptive_avg_pool1d(weather, self.weather_bins).flatten(1)
        weather = F.normalize(self.weather_projection(weather), dim=-1, eps=1e-08).reshape(B, N, self.weather_dim)
        score = weather @ weather.transpose(1, 2) / self.weather_temperature
        score = score + self.prior_max * torch.sigmoid(self.prior_logit) * self.relation_prior
        off_diagonal = ~torch.eye(N, device=state.device, dtype=torch.bool)[None].expand(B, N, N)
        values, indices = score.masked_fill(~off_diagonal, torch.finfo(score.dtype).min).topk(min(self.topk, N - 1), dim=-1)
        valid = off_diagonal.gather(-1, indices)
        A = torch.softmax(values, dim=-1) * valid.to(values.dtype)
        A = A[:, None].expand(B, L, N, indices.shape[-1])
        source_mask = valid[:, None].expand_as(A)
        logits = A.clamp_min(torch.finfo(A.dtype).tiny).log()
        base_logits = logits.masked_fill(~source_mask, torch.finfo(logits.dtype).min)
        base_attention = torch.softmax(base_logits, dim=-1) * source_mask.to(logits.dtype)
        source_error = self.gather(error[..., None], indices).squeeze(-1)
        quality_logits = logits - self.quality_strength * source_error / self.quality_temperature
        quality_logits = quality_logits.masked_fill(~source_mask, torch.finfo(logits.dtype).min)
        quality_attention = torch.softmax(quality_logits, dim=-1) * source_mask.to(logits.dtype)
        attention = torch.where(receiver[..., None], quality_attention, base_attention)
        projected = self.hidden_projection(hidden)
        source_hidden = self.gather(projected, indices)
        source_state, source_base = (self.gather(state[..., None], indices), self.gather(base[..., None], indices))
        source_observed = self.gather(mask[..., None], indices)
        pair = torch.cat((projected.unsqueeze(3).expand_as(source_hidden), source_hidden, self.gather(aux, indices) - aux.unsqueeze(3), source_state - state[..., None, None], source_base - base[..., None, None], source_state - source_base, source_observed, mask[..., None, None].expand_as(source_observed)), dim=-1)
        message = self.message_bound * torch.tanh(self.message_net(pair).squeeze(-1))
        source_mass = (A * source_mask.to(A.dtype)).sum(-1)
        missing_fraction = (mask <= 0.5).to(state.dtype).mean(1, keepdim=True).expand(-1, L, -1)
        zero = torch.zeros_like(state)
        gate_inputs = torch.cat((projected, missing_fraction[..., None], source_mass[..., None], zero[..., None], zero[..., None]), dim=-1)
        gate = torch.sigmoid(self.gate_net(gate_inputs).squeeze(-1))
        delta = gate * (attention * message).sum(-1)
        return {'p_graph': state + delta, 'delta': delta, 'gate': gate, 'attention': attention, 'base_attention': base_attention, 'quality_receiver': receiver, 'neighbor_idx': indices, 'source_error': source_error}


# Complete model and staged optimization
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        cfg = resolve_config(configs)
        cfg.update(model_name='MPRGNetv2',
                   jointq_mode=str(getattr(configs, 'jointq_mode', 'soft')),
                   jointq_scope=str(getattr(configs, 'jointq_scope', 'longgap')),
                   jointq_strength=float(getattr(configs, 'jointq_strength', .5)),
                   jointq_gap_steps=int(getattr(configs, 'jointq_gap_steps', 24)))
        self.resolved_config = cfg
        c = SimpleNamespace(**cfg)
        self.seq_len, self.pred_len = int(c.seq_len), int(c.pred_len)
        self.num_sites = self.target_dim = int(c.target_dim)
        self.enc_in, self.aux_per_site = int(c.enc_in), int(c.aux_per_site)
        self.splitter = InputSplitter(self.num_sites, self.aux_per_site)
        self.restore_head = WeatherPowerStateReconstructionHead(c)
        self.forecast_head = AntiNoiseTemporalForecastHead(c)
        dim = int(c.restore_hidden_dim)
        self.obs_error_head = nn.Sequential(nn.Linear(dim+7, dim), nn.GELU(),
                                            nn.Linear(dim, 1), nn.Softplus())
        self.state_graph = SelectiveStateGraph(c)
        self.state_graph = QualityStateGraph(c)
        self.state_error_head = nn.Sequential(nn.Linear(dim+7, dim), nn.GELU(),
                                              nn.Linear(dim, 1), nn.Softplus())
        nn.init.constant_(self.state_error_head[-2].bias, -2.3)
        self.register_buffer('quality_permutation', torch.randperm(
            self.num_sites, generator=torch.Generator().manual_seed(2718)))
        self.mode, self.intervention = cfg['jointq_mode'], 'real'
        self.window_fraction = cfg['select_window_fraction']
        self.set_stage('joint')

    def set_stage(self, stage):
        if stage not in ('joint', 'quality', 'graph', 'eval'):
            raise ValueError(stage)
        self.stage = stage
        for name, parameter in self.named_parameters():
            if stage == 'joint':
                active = not name.startswith(('obs_error_head.', 'state_error_head.', 'forecast_head.projection.'))
            else:
                prefix = {'quality':'state_error_head.', 'graph':'state_graph.', 'eval':None}[stage]
                active = prefix is not None and name.startswith(prefix)
            parameter.requires_grad_(active)
            parameter.grad = None
        self.train(self.training)
        return self

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if hasattr(self, 'stage'):
            self.restore_head.train(mode and self.stage == 'joint')
            self.forecast_head.train(mode and self.stage == 'joint')
            self.state_graph.train(mode and self.stage in ('joint', 'graph'))
            self.state_error_head.train(mode and self.stage == 'quality')
            self.obs_error_head.eval()
        return self

    def encode(self, x):
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError(f'Expected historical input [B,{self.seq_len},{self.enc_in}]')
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.stage == 'joint'):
            observation, mask, aux = self.splitter(x)
            state, base, alpha, hidden = self.restore_head(observation, mask, aux)
        return {'observation':observation, 'mask':mask, 'aux':aux, 'p_impute':state,
                'p_base':base, 'fusion_alpha':alpha, 'hidden':hidden}

    def quality_outputs(self, features):
        obs, base = features['observation'], features['p_base']
        scalars = torch.stack((obs, base, features['p_impute'], obs-base, (obs-base).abs(),
                               features['mask'], features['fusion_alpha']), dim=-1)
        error = self.state_error_head(torch.cat((features['hidden'], scalars), dim=-1).detach()).squeeze(-1)
        if self.intervention == 'constant':
            error = torch.zeros_like(error)
        elif self.intervention == 'permute':
            error = error.index_select(-1, self.quality_permutation)
        elif self.intervention != 'real':
            raise ValueError(self.intervention)
        count = max(1, math.ceil(self.window_fraction*error.shape[1]))
        window = error.topk(count, dim=1).values.mean(1)
        return {'state_error':error, 'window_error':window, 'q':1/(1+error), 'window_q':1/(1+window)}

    def forward_from_features(self, features):
        values = dict(features)
        if self.mode != 'off':
            values.update(self.quality_outputs(features))
        graph = self.state_graph(values, mode=self.mode)
        return {'pred':self.forecast_head(graph['p_graph']), 'p_impute':features['p_impute'],
                'p_base':features['p_base'], 'fusion_alpha':features['fusion_alpha'], **graph,
                **{k:values[k] for k in ('state_error', 'window_error', 'q', 'window_q') if k in values}}

    def forward(self, x, batch=None):
        return self.forward_from_features(self.encode(x))

    def load_pretrained_core(self, state):
        current = self.state_dict()
        optional = lambda k: k.startswith('state_error_head.') or k == 'quality_permutation'
        expected = {k for k in current if not optional(k)}
        retained = {k:v for k,v in state.items() if not optional(k)}
        if set(retained) != expected:
            raise RuntimeError(f'Core checkpoint mismatch: missing={expected-set(retained)}, unexpected={set(retained)-expected}')
        current.update(retained)
        self.load_state_dict(current, strict=True)

    def set_relation_prior(self, prior, split_name='train'):
        self.state_graph.set_relation_prior(prior, split_name)
        return self


def quality_loss(features, output, batch):
    target = (features['p_impute'].detach()-batch['pv_clean'].detach()).abs()
    return (output['state_error']-target).square().mean()
