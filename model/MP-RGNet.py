import torch
import torch.nn as nn


class InputSplitter(nn.Module):

    def __init__(self, num_sites: int, aux_per_site: int):
        super().__init__()
        self.num_sites = int(num_sites)
        self.aux_per_site = int(aux_per_site)

    def forward(self, x: torch.Tensor):
        # x: [B, L, 2N + N*K] for model_type='ours'
        n = self.num_sites
        k = self.aux_per_site
        p_obs = x[:, :, :n]                      # [B, L, N]
        obs_mask = x[:, :, n:2 * n]              # [B, L, N]
        aux = x[:, :, 2 * n:]                    # [B, L, N*K]
        aux = aux.reshape(x.shape[0], x.shape[1], n, k)  # [B, L, N, K]

        return p_obs, obs_mask, aux

class MeteoPowerTimeVariableFACTBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.hidden_dim = int(configs.restore_hidden_dim)
        self.kernel_time = int(configs.restore_kernel_time)
        self.kernel_var = int(configs.restore_kernel_var)
        self.dilations = tuple(int(d) for d in configs.restore_dilations)
        self.dropout = float(configs.restore_dropout)
        self.branches = nn.ModuleList()
        for d in self.dilations:
            pad_t = d * (self.kernel_time - 1) // 2
            pad_v = d * (self.kernel_var - 1) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        kernel_size=(self.kernel_time, self.kernel_var),
                        padding=(pad_t, pad_v),
                        dilation=(d, d),
                        groups=self.hidden_dim,
                        bias=True,
                    ),
                    nn.GELU(),
                )
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_dim * len(self.branches),
                out_channels=self.hidden_dim,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, power_token: torch.Tensor, swr_token: torch.Tensor):
        # power_token, swr_token: [B, L, N, D]
        B, L, N, D = power_token.shape
        z = torch.stack([power_token, swr_token], dim=3).reshape(B, L, 2 * N, D)
        x = z.permute(0, 3, 1, 2).contiguous()
        h = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        h = h.permute(0, 2, 3, 1).contiguous()
        h = self.norm(h + z)
        h = h.reshape(B, L, N, 2, D)
        return h[:, :, :, 0, :]

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
        self.reliability_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, p_obs: torch.Tensor, obs_mask: torch.Tensor, aux: torch.Tensor):
        # p_obs:    [B, L, N]
        # obs_mask: [B, L, N]
        # aux:      [B, L, N, K]
        B, L, N = p_obs.shape
        site_emb = self.site_embedding.view(1, 1, N, self.hidden_dim)

        power_input = torch.stack([p_obs, obs_mask], dim=-1)
        power_token = self.power_encoder(power_input) + site_emb

        swr = aux[..., self.swr_index:self.swr_index + 1]
        swr_token = self.swr_encoder(swr) + site_emb

        context = power_token
        for block in self.fact_blocks:
            context = block(context, swr_token)

        h = self.fusion(torch.cat([power_token, context], dim=-1))
        p_base = self.p_base_head(h).squeeze(-1)

        innovation = torch.abs(p_obs - p_base).unsqueeze(-1)
        reliability_input = torch.cat([h, innovation, obs_mask.unsqueeze(-1)], dim=-1)
        reliability = torch.sigmoid(self.reliability_head(reliability_input).squeeze(-1))

        p_restore = (
            obs_mask * (reliability * p_obs + (1.0 - reliability) * p_base)
            + (1.0 - obs_mask) * p_base
        )

        return p_restore, p_base, reliability


class WindowMeteoReliabilityGraphSmoother(nn.Module):
    """
    One-window meteorological similarity graph correction.

    Design target:
        - Build one N x N graph per input window, not one graph per time step.
        - Keep only top-k meteorologically similar neighbors.
        - Apply one neighbor weighted average to p_restore.
        - Use reliability only as a conservative residual gate.
        - Do not feed auxiliary features directly to the forecast head.

    Inputs:
        p_restore:   [B, L, N]
        reliability: [B, L, N]
        obs_mask:    [B, L, N]
        aux:         [B, L, N, K]

    Output:
        p_graph:     [B, L, N]
        scale:       scalar tensor
    """

    def __init__(self, configs):
        super().__init__()
        self.num_sites = int(configs.target_dim)
        self.aux_per_site = int(configs.aux_per_site)
        self.swr_index = int(getattr(configs, "swr_index", 0))

        self.topk = int(getattr(configs, "meteo_graph_topk", getattr(configs, "graph_topk", 3)))
        self.max_scale = float(getattr(configs, "meteo_graph_max_scale", getattr(configs, "graph_max_scale", 0.05)))
        self.temperature = float(getattr(configs, "meteo_graph_temperature", getattr(configs, "graph_temperature", 0.5)))
        self.eps = float(getattr(configs, "meteo_graph_eps", getattr(configs, "graph_eps", 1e-6)))
        self.delta_clip = float(getattr(configs, "meteo_graph_delta_clip", getattr(configs, "graph_delta_clip", 20.0)))

        init_logit = float(getattr(configs, "meteo_graph_init_logit", getattr(configs, "graph_init_logit", -8.0)))
        self.gate_logit = nn.Parameter(torch.tensor(init_logit))

        aux_indices = self._parse_aux_indices(configs)
        self.register_buffer("aux_indices", aux_indices, persistent=True)

    def _parse_aux_indices(self, configs) -> torch.Tensor:
        raw = getattr(configs, "meteo_graph_aux_indices", getattr(configs, "graph_aux_indices", None))
        if raw is None:
            indices = list(range(self.aux_per_site))
            exclude_swr = bool(getattr(configs, "meteo_graph_exclude_swr", False))
            if exclude_swr and 0 <= self.swr_index < self.aux_per_site and len(indices) > 1:
                indices.remove(self.swr_index)
        elif isinstance(raw, str):
            indices = [int(item.strip()) for item in raw.split(",") if item.strip() != ""]
        else:
            indices = [int(item) for item in raw]

        indices = [idx for idx in indices if 0 <= idx < self.aux_per_site]
        if len(indices) == 0:
            indices = list(range(self.aux_per_site))

        return torch.tensor(indices, dtype=torch.long)

    def _build_window_features(self, aux: torch.Tensor) -> torch.Tensor:
        # aux_selected: [B, L, N, K_g]
        aux_selected = aux.index_select(dim=-1, index=self.aux_indices)

        # One representation per window and site. Mean captures weather level;
        # std captures intra-window change. Shape: [B, N, 2 * K_g]
        aux_mean = aux_selected.mean(dim=1)
        aux_std = aux_selected.std(dim=1, unbiased=False)
        feat = torch.cat([aux_mean, aux_std], dim=-1)

        # Normalize across sites inside each sample window so that one feature
        # scale does not dominate the meteorological similarity graph.
        feat_mean = feat.mean(dim=1, keepdim=True)
        feat_std = feat.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        feat = (feat - feat_mean) / feat_std

        # L2 normalization makes matmul equivalent to cosine similarity.
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return feat

    def _topk_similarity_graph(self, aux: torch.Tensor):
        B, L, N, K = aux.shape
        if N != self.num_sites:
            raise AssertionError(f"Expected N={self.num_sites}, got {N}.")
        if K != self.aux_per_site:
            raise AssertionError(f"Expected aux_per_site={self.aux_per_site}, got {K}.")

        if N <= 1:
            idx = torch.zeros(B, N, 1, device=aux.device, dtype=torch.long)
            weight = torch.ones(B, N, 1, device=aux.device, dtype=aux.dtype)
            return idx, weight

        feat = self._build_window_features(aux)
        sim = torch.matmul(feat, feat.transpose(1, 2))  # [B, N, N]

        eye = torch.eye(N, device=aux.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, -1e4)

        k = max(1, min(self.topk, N - 1))
        top_score, top_idx = torch.topk(sim, k=k, dim=-1)  # [B, N, k]
        base_weight = torch.softmax(top_score / max(self.temperature, self.eps), dim=-1)
        return top_idx, base_weight

    @staticmethod
    def _gather_neighbors(x: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        # x:            [B, L, N]
        # neighbor_idx: [B, N, k]
        # return:       [B, L, N, k]
        B, L, N = x.shape
        k = neighbor_idx.size(-1)
        x_flat = x.reshape(B * L, N)
        idx_flat = neighbor_idx.unsqueeze(1).expand(B, L, N, k).reshape(B * L, N * k)
        gathered = torch.gather(x_flat, dim=1, index=idx_flat)
        return gathered.reshape(B, L, N, k)

    def forward(
        self,
        p_restore: torch.Tensor,
        reliability: torch.Tensor,
        obs_mask: torch.Tensor,
        aux: torch.Tensor,
    ):
        if p_restore.dim() != 3:
            raise AssertionError(f"p_restore should have shape [B, L, N], got {tuple(p_restore.shape)}.")
        if reliability.shape != p_restore.shape:
            raise AssertionError(
                f"reliability shape {tuple(reliability.shape)} should match p_restore {tuple(p_restore.shape)}."
            )
        if obs_mask.shape != p_restore.shape:
            raise AssertionError(
                f"obs_mask shape {tuple(obs_mask.shape)} should match p_restore {tuple(p_restore.shape)}."
            )
        if aux.shape[:3] != p_restore.shape:
            raise AssertionError(
                f"aux[:3] {tuple(aux.shape[:3])} should match p_restore {tuple(p_restore.shape)}."
            )

        B, L, N = p_restore.shape
        if N <= 1:
            scale = self.max_scale * torch.sigmoid(self.gate_logit)
            return p_restore, scale

        # The graph is built from weather, but the graph itself should not become
        # another learnable path that disturbs the reconstruction head semantics.
        neighbor_idx, base_weight = self._topk_similarity_graph(aux.detach())

        rel = reliability.detach().clamp(0.0, 1.0)
        mask = obs_mask.detach().clamp(0.0, 1.0)

        # Reliable observed sites are better sources. This is window-level so the
        # neighbor graph does not fluctuate at every time step.
        source_quality = (mask * rel).mean(dim=1)  # [B, N]
        src_q = torch.gather(
            source_quality,
            dim=1,
            index=neighbor_idx.reshape(B, -1),
        ).reshape_as(base_weight)

        weighted = base_weight * src_q
        denom = weighted.sum(dim=-1, keepdim=True)
        neighbor_weight = torch.where(
            denom > self.eps,
            weighted / denom.clamp_min(self.eps),
            base_weight,
        )

        src_p = self._gather_neighbors(p_restore.detach(), neighbor_idx)
        p_neigh = (src_p * neighbor_weight.unsqueeze(1)).sum(dim=-1)

        # Low-reliability observed values and missing values receive more graph correction.
        target_uncertainty = 1.0 - mask * rel

        unc_threshold = float(getattr(self, "unc_threshold", 0.5))
        target_uncertainty = torch.where(
            target_uncertainty > unc_threshold,
            target_uncertainty,
            torch.zeros_like(target_uncertainty),
        )

        scale = self.max_scale * torch.sigmoid(self.gate_logit)
        delta = p_neigh - p_restore
        if self.delta_clip > 0:
            delta = self.delta_clip * torch.tanh(delta / self.delta_clip)

        p_graph = p_restore + scale * target_uncertainty * delta
        return p_graph, scale


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AntiNoiseTemporalForecastHead(nn.Module):
    """
    Minimal temporal head kept only to satisfy the existing training interface,
    which requires model_output['pred'].
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.hidden_size = int(configs.hidden_size)
        self.decompsition = series_decomp(25)
        self.projection = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.GELU(),
            nn.Linear(self.seq_len, self.pred_len),
        )
        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )
    def forward(self, p_restore: torch.Tensor):
        # p_restore: [B, L, N]
        # y_hat:     [B, pred_len, N]
        seasonal, trend = self.decompsition(p_restore)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out = seasonal + trend
        return out

class Model(nn.Module):
    """
    Restore-first wrapper.

    Expected input:
        x = [pv_obs_N, obs_mask_N, aux_N*K]

    Returned tensors:
        pred:        [B, pred_len, N]
        p_impute:    [B, seq_len, N]
        p_base:      [B, seq_len, N]
        reliability: [B, seq_len, N]
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.num_sites = int(configs.target_dim)
        self.target_dim = int(configs.target_dim)
        self.enc_in = int(configs.enc_in)
        self.aux_per_site = int(configs.aux_per_site)

        self.splitter = InputSplitter(
            num_sites=self.num_sites,
            aux_per_site=self.aux_per_site,
        )
        self.restore_head = WeatherPowerStateReconstructionHead(configs)
        self.use_meteo_graph = bool(getattr(configs, "use_meteo_graph", True))
        if self.use_meteo_graph:
            self.meteo_graph_smoother = WindowMeteoReliabilityGraphSmoother(configs)
        else:
            self.meteo_graph_smoother = None
        self.forecast_head = AntiNoiseTemporalForecastHead(configs)

    def forward(self, x: torch.Tensor, batch=None):
        p_obs, obs_mask, aux = self.splitter(x)
        p_restore, p_base, reliability = self.restore_head(
            p_obs=p_obs,
            obs_mask=obs_mask,
            aux=aux,
        )

        if self.use_meteo_graph:
            p_graph, graph_scale = self.meteo_graph_smoother(
                p_restore=p_restore,
                reliability=reliability,
                obs_mask=obs_mask,
                aux=aux,
            )
        else:
            p_graph = p_restore
            graph_scale = p_restore.new_tensor(0.0)

        y_hat = self.forecast_head(p_graph)

        return {
            "pred": y_hat,
            "p_impute": p_restore,
            "p_graph": p_graph,
            "p_base": p_base,
            "reliability": reliability,
        }
