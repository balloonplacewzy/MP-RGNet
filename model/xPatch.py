import torch
import torch.nn as nn
import math
from layers.decomp import DECOMP
from layers.network import Network
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Normalization
        self.revin = True
        self.revin_layer = RevIN(configs.enc_in, affine=False,subtract_last=False)
        self.target_dim = configs.target_dim
        # Moving Average
        self.ma_type = "ema"
        alpha = 0.5      # smoothing factor for EMA (Exponential Moving Average)
        beta = 0.3         # smoothing factor for DEMA (Double Exponential Moving Average)
        self.patch_len = configs.patch_len
        seq_len, pred_len, patch_len, stride, padding_patch = configs.seq_len, configs.pred_len, configs.patch_len, configs.stride, configs.padding_patch
        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream


    def forward(self, x):
        # x: [Batch, Input, Channel]


        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)


        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x[:,:,:self.target_dim]