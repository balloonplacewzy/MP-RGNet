# -*- coding: utf-8 -*-

class BaseFullConfig:
    def __init__(self):
        # ---------------- data ----------------
        self.data = r"./data_provider/pv_power_selected_aux_7features.csv"
        self.num_sites = 227
        self.target_dim = 227
        self.enc_out = 227
        self.enc_in = 1816  # 227 PV + 7 * 227 auxiliary features
        # ---------------- data-quality experiment ----------------
        # Supported scenarios:
        #   clean
        #   noise_01, noise_02, noise_03
        #   missing_10, missing_30, missing_50
        #   mixed_10, mixed_20, mixed_30
        # Shared scenario for all models. Explicit CLI arguments can override it.
        self.scenario_name = "mixed_10"
        self.missing_pattern = "random_point"  # "block","station","random_point"
        self.block_len = 12
        self.aux_delay_steps = 4
        self.data_seed = 2026
        self.model_type = "baseline"
        self.impute_method = "forward_fill"
        # Original CSV auxiliary layout:
        #   feature_major means aux_0_site_0..aux_0_site_226,
        #   aux_1_site_0..aux_1_site_226, ...
        # The dataloader converts auxiliary columns to unit-major before
        # concatenating them into x, so models can reshape aux as [B, L, N, 7].
        self.aux_layout = "feature_major"
        # None retains the mask-input default for each model type.
        self.append_obs_mask = None
        # KNN is used only when model_type == "baseline" and the scenario has
        # missing PV observations. These values are ignored otherwise.
        self.knn_n_neighbors = 5
        self.knn_weights = "distance"
        # ---------------- optimization ----------------
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 50
        self.patience = 8
        self.decay_patience = 3
        self.grad_clip = 1.0
        self.loss = "mae"
        # ---------------- runtime ----------------
        self.seed = 2026
        self.device = "cuda"
        self.num_workers = 0
        self.use_amp = False
        self.drop_last = True
        # ---------------- split / preprocessing ----------------
        self.split_ratio = (0.8, 0.1, 0.1)
        self.scale = True
        # --------------- saving ----------------
        self.checkpoint_root = "./checkpoint_quality"
        self.result_root = "./result_quality"
        # ---------------- evaluation ----------------
        self.day_threshold = 1e-6
        self.timing_warmup_batches = 5
        self.timing_measure_batches = 30
        # ---------------- auxiliary losses for proposed models ----------------
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

        # Data
        self.seq_len = 96
        self.label_len = 0
        self.pred_len = 48

        # Baseline input: [PV_filled_227, aux_7*227] = 1816
        self.enc_in = 1816
        self.dec_in = 1816
        self.c_out = 1816
        self.target_dim = 227
        self.enc_out = 227

        # TimeMixer core
        self.channel_independence = 0
        self.d_model = 128
        self.d_ff = 256
        self.e_layers = 2
        self.dropout = 0.20

        # Decomposition
        self.decomp_method = "moving_avg"
        self.moving_avg = 25
        self.top_k = 5

        # Multi-scale
        self.down_sampling_method = "avg"
        self.down_sampling_window = 2
        self.down_sampling_layers = 1

        # Normalization / embedding
        self.use_norm = 1
        self.embed = "timeF"
        self.freq = "h"
        self.use_future_temporal_feature = False

        # Optimization
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

class FACTConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "FACT"
        self.task_name = "long_term_forecast"
        self.model_type = "baseline"
        self.impute_method = "auto"
        self.features = "M"
        # Baseline input: [PV_filled_227, aux_7*227] = 1816.
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
        # Optimization.
        self.lr = 8e-4
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 100
        self.patience = 10
        self.decay_patience = 4
        self.grad_clip = 1.0
        self.loss = "mae"
        # No auxiliary losses for baseline models.
        self.lambda_impute = 0.0
        self.lambda_unc = 0.0
        self.lambda_smooth = 0.0
        self.lambda_graph = 0.0

#
# class MeteoAmplifierConfig(BaseFullConfig):
#     def __init__(self):
#         super().__init__()
#         self.model_name = "MeteoAmplifier"
#         self.task_name = "long_term_forecast"
#         self.model_type = "ours"
#         self.impute_method = "auto"
#
#         # External input:
#         # [pv_obs_N, obs_mask_N, aux_7N] = 227 * 9 = 2043
#         self.seq_len = 96
#         self.pred_len = 8
#         self.feature_num = 9
#         self.enc_in = self.target_dim * self.feature_num
#         self.enc_out = self.target_dim
#         self.aux_layout = "feature_major"
#
#         # MeteoPowerImputer
#         self.d_model = 64
#         self.imputer_d_model = 64
#         self.imputer_dropout = 0.1
#         self.imputer_obs_fuse_logit_init = 2.0
#         self.imputer_miss_quality_logit_init = 0.0
#
#         self.hidden_size = 128
#         self.moving_avg = 13
#         self.use_freq_branch = True
#         self.freq_hidden_size = 128
#         self.freq_dropout = 0.0
#         self.freq_gate_init = -3.0
#
#         # Auxiliary p_local supervision
#         self.return_model_extras = False
#         self.return_aux_losses = True
#         self.impute_loss_on_clean = True
#         self.lambda_impute = 0.5
#         self.lambda_unc = 0.0
#         self.lambda_smooth = 0.0
#         self.lambda_graph = 0.0
#
#         # Optimization
#         self.lr = 8e-4
#         self.weight_decay = 1e-5
#         self.batch_size = 64
#         self.epochs = 100
#         self.patience = 10
#         self.decay_patience = 4
#         self.grad_clip = 1.0
#         self.loss = "mae"
#         self.dropout = 0.1



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

        # Sequence settings.
        self.seq_len = 96
        self.pred_len = 48

        # Data dimensions.
        self.target_dim = 227
        self.enc_out = 227
        self.aux_per_site = 7
        self.feature_num = 9                  # pv_obs + obs_mask + 7 aux = 9 features per site
        self.enc_in = self.target_dim * self.feature_num
        self.aux_layout = "feature_major"
        self.swr_index = 0

        # Optimization.
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 100
        self.patience = 16
        self.decay_patience = 5
        self.grad_clip = 1.0
        self.loss = "mae"
        self.dropout = 0.1

        # Forecast head.
        self.hidden_size = 128

        # Reconstruction head.
        self.restore_hidden_dim = 64
        self.restore_fact_layers = 1
        self.restore_kernel_time = 5
        self.restore_kernel_var = 5
        self.restore_dilations = (1, 2, 4)
        self.restore_dropout = 0.1
        self.restore_site_emb_scale = 0.02

        # Window meteorological reliability graph.
        self.use_meteo_graph = False

        # All auxiliary features are used to build the meteorological similarity graph.
        # For aux_per_site=7, this means indices 0,1,2,3,4,5,6.
        self.meteo_graph_aux_indices = tuple(range(self.aux_per_site))
        self.meteo_graph_exclude_swr = False

        # Keep the graph small and conservative.
        self.meteo_graph_topk = 3
        self.meteo_graph_max_scale = 0.01
        self.meteo_graph_temperature = 0.8
        self.meteo_graph_eps = 1e-6
        self.meteo_graph_delta_clip = 2.0
        self.meteo_graph_topk = 3
        self.meteo_graph_init_logit = -8.0

        # Training loss controls.
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



class MPRGNetv2Config(newmodelAUXGRAPHConfig):
    def __init__(self):
        super().__init__()
        # Data and runtime
        self.model_name = "MPRGNetv2"
        self.scenario_name = "mixed_30"
        self.seq_len = 96
        self.pred_len = 8
        self.append_obs_mask = True
        self.impute_method = "zero"
        self.batch_size = 16
        self.drop_last = False
        self.restore_dropout = 0.05
        self.seeds = [2026, 2027, 2028]
        self.initialization = "pretrained"
        self.initial_checkpoint_dir = "research/qselect/confirm_noq_20260930"
        self.evaluation_checkpoint_dir = "research/joint_quality/confirm_softq_20261002"
        self.output_root = "mprgnetv2_results"
        self.num_threads = 4
        # Joint prediction training
        self.lr = 1e-4
        self.joint_graph_lr = 3e-4
        self.epochs = 8
        self.patience = 4
        self.decay_patience = 2
        self.lambda_impute = 0.0
        self.train_windows = 0
        self.validation_windows = 0
        # Quality calibration and graph adaptation
        self.quality_epochs = 6
        self.quality_lr = 1e-3
        self.quality_windows = 2048
        self.quality_validation_windows = 192
        self.graph_epochs = 6
        self.graph_lr = 1e-4
        self.graph_windows = 2048
        self.graph_guard_weight = 0.5
        self.random_tolerance = 0.005
        # Graph structure and bounded correction
        self.select_weather_bins = 8
        self.select_weather_dim = 16
        self.select_hidden_dim = 16
        self.select_message_hidden = 32
        self.select_gate_hidden = 32
        self.select_topk = 8
        self.select_weather_temperature = 0.5
        self.select_prior_max = 2.0
        self.select_quality_temperature = 0.15
        self.select_window_fraction = 0.25
        self.select_receiver_fraction = 0.5
        self.select_window_threshold = 0.10
        self.select_point_threshold = 0.075
        self.select_source_threshold = 0.15
        self.select_margin = 0.025
        self.select_message_bound = 0.5
        self.select_gate_init_bias = -1.0
        self.jointq_mode = "soft"
        self.jointq_scope = "longgap"
        self.jointq_strength = 0.5
        self.jointq_gap_steps = 24


class MPRGNetv2SingleSceneConfig(MPRGNetv2Config):
    def __init__(self):
        super().__init__()
        # Single-scene continuation from original backbone checkpoints
        self.initialization = "single_scene_backbone"
        self.initial_checkpoint_dir = None
        self.backbone_checkpoint_template = "checkpoint_quality/pv_power_selected_aux_7features/{scene}/newmodelAUXGRAPH_type-ours_stage-joint_scenario-{scene}_imp-zero_len{seq_len}-{pred_len}_seed{seed}.pth"
        self.mixed30_backbone_checkpoint_template = "result_quality/mixed3_h8_three_seeds/newmodelAUXGRAPH_seed{seed}/selected.pth"
        self.scenarios = ["mixed_10", "mixed_20", "mixed_30"]
        self.seeds = [2026]
        self.output_root = "research/mprgnetv2_single_scene"
        self.warmup_epochs = 8
        self.warmup_windows = 1536
        self.warmup_lr = 1e-3
        self.warmup_history_weight = 0.05


class GraphWaveNetConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        # Data and prediction
        self.model_name = "GraphWaveNet"
        self.task_name = "long_term_forecast"
        self.seq_len = 96
        self.pred_len = 8
        self.aux_per_site = 7
        self.impute_method = "knn"
        self.append_obs_mask = False
        # Adaptive-only Graph WaveNet; automatic depth covers the full history
        self.gwn_graph = "adaptive_only"
        self.gwn_blocks = 1
        self.gwn_layers = None
        self.gwn_kernel_size = 2
        self.gwn_node_dim = 10
        self.gwn_order = 2
        # Compact channel widths
        self.gwn_residual_channels = 16
        self.gwn_dilation_channels = 16
        self.gwn_skip_channels = 64
        self.gwn_end_channels = 128
        self.dropout = 0.3
        # Optimization
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 64
        self.epochs = 100
        self.patience = 6
        self.decay_patience = 3
        self.grad_clip = 5.0
        self.loss = "mae"


class HD_TTSConfig(BaseFullConfig):
    def __init__(self):
        super().__init__()
        self.model_name = "HD_TTS"
        self.task_name = "long_term_forecast"
        self.seq_len = 96
        self.pred_len = 8
        self.aux_per_site = 7
        self.enc_in = self.num_sites * (2 + self.aux_per_site)
        self.impute_method = "zero"
        self.append_obs_mask = True
        self.hdtts_coords = "data_provider/processed/coords_227.pt"
        self.hdtts_graph_k = 8
        self.hdtts_graph_threshold = 0.1
        self.hdtts_hidden = 64
        # Same optimization/selection protocol as the comparison entry.
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 32
        self.epochs = 100
        self.patience = 6
        self.decay_patience = 3
        self.grad_clip = 5.0
        self.loss = "mae"


def get_config(model_name: str):
    if model_name == "newmodelAUXGRAPH":
        return newmodelAUXGRAPHConfig()
    if model_name in ("HD_TTS", "HD-TTS", "HDTTS"):
        return HD_TTSConfig()
    if model_name == "GraphWaveNet":
        return GraphWaveNetConfig()
    if model_name == "MPRGNetv2":
        return MPRGNetv2Config()
    if model_name == "MPRGNetv2SingleScene":
        return MPRGNetv2SingleSceneConfig()
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
