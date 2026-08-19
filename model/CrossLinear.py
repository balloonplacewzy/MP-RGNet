import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Patch_Embedding(nn.Module):
    """
    Input : [B, C, L]
    Output: [B, C, patch_num, d_model]
    """

    def __init__(self, seq_len, patch_num, patch_len, d_model, d_ff, variate_num):
        super(Patch_Embedding, self).__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len

        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, C, L]
        if self.pad_num > 0:
            x = F.pad(x, (0, self.pad_num))
        x = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        x = self.linear(x)
        return x


class De_Patch_Embedding(nn.Module):
    """
    Input : [B, C, patch_num, d_model]
    Output: [B, C, pred_len]
    """

    def __init__(self, pred_len, patch_num, d_model, d_ff, variate_num):
        super(De_Patch_Embedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, x):
        return self.linear(x)


class Model(nn.Module):
    """
    CrossLinear adapted for your Stage-1 framework.

    Key design:
    1. Use configs.enc_in as the real input feature dimension.
       - No-aux version: args.enc_in = args.target_dim
       - Aux version   : args.enc_in = args.total_features
    2. Use configs.target_dim as the final forecast dimension.
       - The first target_dim columns are treated as target variables.
    3. Ignore configs.dec_in and the old MS logic.
       - The original CrossLinear used configs.dec_in as input channels,
         which conflicts with your two main_stage1 files.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = getattr(configs, "task_name", "short_term_forecast")
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.input_dim = int(configs.enc_in)
        self.target_dim = int(configs.target_dim)
        self.EPS = 1e-5

        if self.target_dim > self.input_dim:
            raise ValueError(
                f"target_dim ({self.target_dim}) cannot be larger than enc_in/input_dim ({self.input_dim}). "
                f"Please check your config."
            )

        patch_len = int(configs.patch_len)
        if patch_len <= 0:
            raise ValueError(f"patch_len must be positive, but got {patch_len}.")

        patch_num = math.ceil(self.seq_len / patch_len)
        variate_num = self.input_dim

        alpha = float(getattr(configs, "alpha", 0.5))
        beta = float(getattr(configs, "beta", 0.5))

        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.beta = nn.Parameter(torch.ones(1) * beta)

        # Cross-variable correlation branch.
        # Input and output channels are both enc_in, so it works for both:
        # [target only] and [target + auxiliary features].
        self.correlation_embedding = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=variate_num,
            kernel_size=3,
            padding="same",
        )

        # Patch value branch.
        self.value_embedding = Patch_Embedding(
            seq_len=self.seq_len,
            patch_num=patch_num,
            patch_len=patch_len,
            d_model=int(configs.d_model),
            d_ff=int(configs.d_ff),
            variate_num=variate_num,
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, variate_num, patch_num, int(configs.d_model))
        )

        self.head = De_Patch_Embedding(
            pred_len=self.pred_len,
            patch_num=patch_num,
            d_model=int(configs.d_model),
            d_ff=int(configs.d_ff),
            variate_num=variate_num,
        )

    def forecast(self, x_enc):
        """
        Args:
            x_enc: [B, seq_len, enc_in]
        Returns:
            y_out: [B, pred_len, target_dim]
        """
        if x_enc.ndim != 3:
            raise ValueError(
                f"CrossLinear expects input shape [B, seq_len, enc_in], but got {tuple(x_enc.shape)}."
            )

        if x_enc.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input feature dim mismatch: model was built with enc_in={self.input_dim}, "
                f"but received x_enc.shape[-1]={x_enc.shape[-1]}. "
                f"For no-aux training, set args.enc_in=args.target_dim; "
                f"for aux training, set args.enc_in=args.total_features."
            )

        # [B, L, C] -> [B, C, L]
        x_enc = x_enc.permute(0, 2, 1)

        # Per-variable normalization over time.
        mean = x_enc.mean(dim=-1, keepdim=True)
        std = x_enc.std(dim=-1, keepdim=True, unbiased=False).clamp_min(self.EPS)
        x_norm = (x_enc - mean) / std

        # Correlation-enhanced value representation.
        corr = self.correlation_embedding(x_norm)
        x_obj = self.alpha * x_norm + (1.0 - self.alpha) * corr

        # Patch embedding + position mixing.
        value_emb = self.value_embedding(x_obj)
        x_obj = self.beta * value_emb + (1.0 - self.beta) * self.pos_embedding

        # [B, C, pred_len]
        y_out = self.head(x_obj)

        # De-normalize each variable, then keep only target variables.
        y_out = y_out * std + mean
        y_out = y_out.permute(0, 2, 1)  # [B, pred_len, C]
        y_out = y_out[:, :, : self.target_dim]

        return y_out

    def forward(self, x_enc):
        return self.forecast(x_enc)
