import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Safe iTransformer version for Stage-2 rollout training.
    Main fixes:
    1) remove inplace normalization ops (e.g. x_enc /= stdev)
    2) clone/contiguous the input to avoid view-version conflicts
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.target_dim = configs.target_dim
        self.output_attention = configs.output_attention

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def _safe_norm(self, x_enc):
        # break possible view relationship from rollout slices
        x_enc = x_enc.clone().contiguous()

        means = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = centered / stdev
        return x_norm, means, stdev

    def _safe_denorm(self, dec_out, means, stdev, out_len):
        scale = stdev[:, 0:1, :].expand(-1, out_len, -1)
        bias = means[:, 0:1, :].expand(-1, out_len, -1)
        return dec_out * scale + bias

    def forecast(self, x_enc, x_mark_enc=None):
        x_norm, means, stdev = self._safe_norm(x_enc)
        _, _, n_vars = x_norm.shape

        enc_out = self.enc_embedding(x_norm, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = self._safe_denorm(dec_out, means, stdev, self.pred_len)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_norm, means, stdev = self._safe_norm(x_enc)
        _, seq_len, n_vars = x_norm.shape

        enc_out = self.enc_embedding(x_norm, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = self._safe_denorm(dec_out, means, stdev, seq_len)
        return dec_out

    def anomaly_detection(self, x_enc):
        x_norm, means, stdev = self._safe_norm(x_enc)
        _, seq_len, n_vars = x_norm.shape

        enc_out = self.enc_embedding(x_norm, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = self._safe_denorm(dec_out, means, stdev, seq_len)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc = x_enc.clone().contiguous()
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc)
            return dec_out[:, -self.pred_len:, :self.target_dim]
        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
