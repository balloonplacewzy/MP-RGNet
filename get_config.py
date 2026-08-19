

class BaseFullConfig:
    def __init__(self):

        self.data = r"./data_provider/pv_power_selected_aux_7features.csv"
        self.num_sites = 227
        self.target_dim = 227
        self.enc_out = 227
        self.enc_in = 1816


        self.scenario_name = "mixed_10"
        self.model_type = "baseline"
        self.impute_method = "forward_fill"


        self.aux_layout = "feature_major"


        self.knn_n_neighbors = 5
        self.knn_weights = "distance"

        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 50
        self.patience = 8
        self.decay_patience = 3
        self.grad_clip = 1.0
        self.loss = "mae"

        self.seed = 2026
        self.device = "cuda"
        self.num_workers = 0
        self.use_amp = False
        self.drop_last = True

        self.split_ratio = (0.8, 0.1, 0.1)
        self.scale = True

        self.checkpoint_root = "./checkpoint_quality"
        self.result_root = "./result_quality"

        self.day_threshold = 1e-6
        self.timing_warmup_batches = 5
        self.timing_measure_batches = 30

        self.lambda_impute = 0.0
        self.lambda_unc = 0.0
        self.lambda_smooth = 0.0
        self.lambda_graph = 0.0

class TimeMixerConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "TimeMixer"
        self.task_name = "long_term_forecast"
        self.model_type = "baseline"
        self.impute_method = "auto"


        self.seq_len = 96
        self.label_len = 0
        self.pred_len = 48


        self.enc_in = 1816
        self.dec_in = 1816
        self.c_out = 1816
        self.target_dim = 227
        self.enc_out = 227


        self.channel_independence = 0
        self.d_model = 128
        self.d_ff = 256
        self.e_layers = 2
        self.dropout = 0.20


        self.decomp_method = "moving_avg"
        self.moving_avg = 25
        self.top_k = 5


        self.down_sampling_method = "avg"
        self.down_sampling_window = 2
        self.down_sampling_layers = 1


        self.use_norm = 1
        self.embed = "timeF"
        self.freq = "h"
        self.use_future_temporal_feature = False


        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.batch_size = 64
        self.epochs = 100
        self.patience = 12
        self.decay_patience = 3
        self.grad_clip = 0.5
        self.loss = "mae"

        self.lambda_impute = 0.0
        self.lambda_unc = 0.0
        self.lambda_smooth = 0.0
        self.lambda_graph = 0.0


class AmplifierConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "Amplifier1"
        self.task_name = "long_term_forecast"
        self.model_type = "baseline"
        self.seq_len = 96
        self.pred_len = 8
        self.enc_in = 1816
        self.enc_out = 227
        self.hidden_size = 64
        self.individual = False
        self.SCI = True
        self.lr = 0.0008
        self.epochs = 100
        self.patience = 8
        self.decay_patience = 4


class DLinearConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "DLinear"
        self.task_name = "short_term_forecast"
        self.model_type = "baseline"
        self.scenario_name = "mixed_10"
        self.individual = False
        self.enc_in = 1816
        self.target_dim = 227
        self.enc_out = 227
        self.moving_avg = 25
        self.seq_len = 96
        self.pred_len = 8
        self.lr = 0.0002
        self.epochs = 1

class CrossLinearConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "CrossLinear"
        self.task_name = "short_term_forecast"
        self.features = "M"
        self.target_dim = 227
        self.enc_in = 1816
        self.enc_out = 227
        self.seq_len = 96
        self.pred_len = 48
        self.patch_len = 12
        self.d_model = 64
        self.d_ff = 128
        self.alpha = 0.8
        self.beta = 0.2
        self.lr = 2e-4
        self.epochs = 100
        self.loss = "mae"
        self.weight_decay = 0.0
        self.patience = 10
        self.decay_patience = 3
        self.grad_clip = 0.0
        self.model_type = "baseline"
        self.impute_method = "auto"
        self.scenario_name = "mixed_10"


class XLinearConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "XLinear"
        self.task_name = "long_term_forecast"
        self.model_type = "baseline"
        self.features = "M"
        self.seq_len = 96
        self.pred_len = 48
        self.enc_in = 1816
        self.target_dim = 227
        self.aux_per_site = 7
        self.site_exogenous = True
        self.d_model = 64
        self.t_ff = 128
        self.c_ff = 32
        self.usenorm = True
        self.embed_dropout = 0.1
        self.head_dropout = 0.1
        self.t_dropout = 0.1
        self.c_dropout = 0.1
        self.epochs = 100
        self.patience = 12

class xPatchConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "xPatch"
        self.task_name = "short_term_forecast"
        self.features = "M"
        self.target_dim = 227
        self.enc_in = 1816
        self.enc_out = 227
        self.seq_len = 96
        self.pred_len = 48
        self.patch_len = 12
        self.stride = 4
        self.padding_patch = "end"
        self.revin = True
        self.ma_type = "ema"
        self.lr = 8e-4
        self.batch_size = 64
        self.epochs = 100
        self.loss = "mae"
        self.patience = 10
        self.decay_patience = 3
        self.grad_clip = 0.0
        self.model_type = "baseline"
        self.impute_method = "auto"
        self.scenario_name = "mixed_10"

class FACTConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "FACT"
        self.task_name = "long_term_forecast"
        self.model_type = "baseline"
        self.impute_method = "auto"
        self.features = "M"

        self.seq_len = 96
        self.pred_len = 8
        self.enc_in = 1816
        self.dec_in = 1816
        self.c_out = 1816
        self.target_dim = 227
        self.enc_out = 227
        self.aux_layout = "feature_major"
        self.freq = "none"
        self.use_norm = True
        self.d_model = 128
        self.d_ff = 256
        self.dropout = 0.1
        self.num_kernels = 3
        self.dilation = [1]


        self.core = 1

        self.lr = 8e-4
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 100
        self.patience = 10
        self.decay_patience = 4
        self.grad_clip = 1.0
        self.loss = "mae"

        self.lambda_impute = 0.0
        self.lambda_unc = 0.0
        self.lambda_smooth = 0.0
        self.lambda_graph = 0.0


class iTransformerConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.task_name = 'short_term_forecast'
        self.model_name = "iTransformer"
        self.model_type = "baseline"
        self.individual = False
        self.output_attention = False
        self.features = 'M'
        self.use_norm = True
        self.embed = True
        self.activation = True
        self.patch_len = 48
        self.enc_in = 1816
        self.d_model = 128
        self.n_heads = 4
        self.e_layers = 2
        self.d_ff = 256
        self.factor = 3
        self.freq = 'h'
        self.seq_len = 96
        self.pred_len = 8
        self.target_dim = 227
        self.dropout = 0.2
        self.lr = 0.001
        self.batchsize = 64
        self.epochs = 50

class SegRNNConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.task_name = 'short_term_forecast'
        self.model_name = "SegRNN"
        self.model_type = "baseline"
        self.hidden_size = 256
        self.individual = False
        self.seq_len = 96
        self.pred_len = 8
        self.seg_len = 8
        self.enc_in = 1816
        self.target_dim = 227
        self.d_model = 128
        self.dropout = 0.2
        self.lr = 0.0008
        self.batchsize = 64
        self.epochs = 100

class TimesNetConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "TimesNet"
        self.task_name = "long_term_forecast"
        self.model_type = "baseline"
        self.impute_method = "auto"

        self.seq_len = 96
        self.label_len = 0
        self.pred_len = 8

        self.enc_in = 1816
        self.dec_in = 1816
        self.c_out = 227
        self.target_dim = 227
        self.enc_out = 227

        self.d_model = 64
        self.d_ff = 128
        self.e_layers = 2
        self.top_k = 3
        self.num_kernels = 3
        self.dropout = 0.2

        self.embed = "timeF"
        self.freq = "h"

        self.lr = 4e-4
        self.batch_size = 64
        self.epochs = 100
        self.patience = 10
        self.decay_patience = 4
        self.grad_clip = 1.0
        self.loss = "mae"


class newmodelAUXGRAPHConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()

        self.model_name = "newmodelAUXGRAPH"
        self.task_name = "long_term_forecast"
        self.model_type = "ours"
        self.impute_method = "auto"


        self.seq_len = 96
        self.pred_len = 48


        self.target_dim = 227
        self.enc_out = 227
        self.aux_per_site = 7
        self.feature_num = 9
        self.enc_in = self.target_dim * self.feature_num
        self.aux_layout = "feature_major"
        self.swr_index = 0


        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 100
        self.patience = 16
        self.decay_patience = 5
        self.grad_clip = 1.0
        self.loss = "mae"
        self.dropout = 0.1


        self.hidden_size = 128


        self.restore_hidden_dim = 64
        self.restore_fact_layers = 1
        self.restore_kernel_time = 5
        self.restore_kernel_var = 5
        self.restore_dilations = (1, 2, 4)
        self.restore_dropout = 0.1
        self.restore_site_emb_scale = 0.02


        self.use_meteo_graph = False


        self.meteo_graph_aux_indices = tuple(range(self.aux_per_site))
        self.meteo_graph_exclude_swr = False


        self.meteo_graph_topk = 3
        self.meteo_graph_max_scale = 0.01
        self.meteo_graph_temperature = 0.8
        self.meteo_graph_eps = 1e-6
        self.meteo_graph_delta_clip = 2.0
        self.meteo_graph_topk = 3
        self.meteo_graph_init_logit = -8.0


        self.restore_only = False
        self.detach_pred_for_restore = True
        self.impute_loss_mode = "all"
        self.return_aux_losses = True
        self.impute_loss_on_clean = True
        self.lambda_pred = 2.0
        self.lambda_impute = 1.0
        self.lambda_unc = 0.0
        self.lambda_smooth = 0.0
        self.lambda_graph = 0.0

def get_config(model_name: str):
    if model_name == "newmodelAUXGRAPH":
        return newmodelAUXGRAPHConfig()
    if model_name == "Amplifier1":
        return AmplifierConfig()
    if model_name == "TimeMixer":
        return TimeMixerConfig()
    if model_name == "DLinear":
        return DLinearConfig()
    if model_name == "CrossLinear":
        return CrossLinearConfig()
    if model_name == "XLinear":
        return XLinearConfig()
    if model_name == "xPatch":
        return xPatchConfig()
    if model_name == "FACT":
        return FACTConfig()
    if model_name == "SegRNN":
        return SegRNNConfig()
    if model_name == "TimesNet":
        return TimesNetConfig()
    raise ValueError(f"Unknown model_name: {model_name}. Please add its Config class in get_config.py")
