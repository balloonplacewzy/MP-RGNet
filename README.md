# MP-RGNet

Code and single-seed checkpoints for **Meteo–Power Reconstruction and State-Quality-Guided Graph Network for Multi-Site Photovoltaic Power Forecasting under Incomplete Information**.

## Installation

Tested with Python 3.12 and PyTorch 2.3.1. Run the commands below from the repository root.

```bash
git clone https://github.com/balloonplacewzy/MP-RGNet.git
cd MP-RGNet
python -m pip install -r requirements.txt
```

The fine-tuning and evaluation commands use CUDA by default. Pass `--device cpu` to run them on a CPU.

## Data and files

The entry points automatically extract `dataset.zip` to `data/pv_power_selected_aux_7features.csv` on first use. The dataset contains 227 PV sites at 15-minute intervals. A 96-step history (24 h) is used to predict 8, 16, 24, or 48 steps (2, 4, 6, or 12 h). The chronological training/validation/test split is 80%/10%/10%, with normalization fitted on the training set.

| Scenario | Paper label | Power missing rate | Gaussian noise std. |
|---|---|---:|---:|
| `mixed_10` | Mixed-1 | 10% | 0.1 |
| `mixed_20` | Mixed-2 | 20% | 0.2 |
| `mixed_30` | Mixed-3 | 30% | 0.3 |

Noise is applied to daytime historical power in standardized units. Missing values are sampled at individual points. Meteorological inputs have a four-step (1 h) delay in these scenarios.

Each `runs/h{horizon}/{scenario}/` directory contains:

| File | Purpose |
|---|---|
| `backbone.pth` | Pretrained reconstruction and forecasting components used to initialize fine-tuning. |
| `model.pt` | Final MP-RGNet checkpoint for direct evaluation. |
| `model_config.json` | Model and data configuration for this condition. |
| `training_config.json` | Stage-specific fine-tuning settings for this condition. |

Model implementations are in `model/`; data loading and scenario generation are in `Dataprovider/`. The `main_mprgnetv2_*.py` files contain the staged training routines used by `train_single_seed.py`.

## Evaluate a supplied checkpoint

```bash
python reproduce.py evaluate --run h8/mixed_10 --device cuda --output results/evaluation_h8_mixed10.json
```

Replace `--run h8/mixed_10` with `--all` to evaluate all 12 conditions. The output JSON contains MAE, RMSE, WAPE, and R2. To evaluate a newly trained model, add `--checkpoint PATH_TO_BEST_PT` to a single-condition command.

## Train from a supplied backbone

```bash
python train_single_seed.py --horizon 8 --scenario mixed_10 --seed 2026 --device cuda --output results/train_h8_mixed10
```

This starts from `backbone.pth` and runs graph warm-up, the configured joint optimization, quality calibration, and graph adaptation. The joint optimization recipe depends on the horizon and scenario; some conditions retain the warm-up result without additional joint updates. Stage-specific validation rules determine the retained checkpoints. Settings are read from the two JSON files in the corresponding `runs/` directory. The final checkpoint is saved as `best.pt` in the output directory. Choose a new output directory for each run.

Fine-tuning seeds share the pretrained backbone for their scenario–horizon condition and use `data_seed=2026`. Graph warm-up freezes the pretrained core; quality calibration updates the state-error estimator; graph adaptation updates the graph module while keeping the other components fixed. The per-condition files specify the training budgets and joint optimization recipe. Validation selection still applies to checkpoints and, for H24/H48, to three quality-reweighting configurations: `(longgap, 0.5)`, `(all, 0.1)`, and `(all, 0.5)`. In `longgap` mode, quality reweighting is restricted to sites with a missing run of at least 24 steps (6 h). Final quality settings are saved in each checkpoint.

Available horizons are `8`, `16`, `24`, and `48`; scenarios are `mixed_10`, `mixed_20`, and `mixed_30`. Add `--smoke` for a short execution check using small subsets and reduced epochs.

## Train a new backbone

The original reconstruction/forecasting training entry is also included.

The checkpoint is saved under `checkpoint_quality/`. Pass its path as `--backbone PATH_TO_CHECKPOINT` to the fine-tuning command above. Use a backbone trained for the same horizon and scenario.

`get_config.py` retains configuration defaults from the local experiments and provides a reference for backbone training. Its data, checkpoint, and output paths follow the original local project layout and can be adjusted for another environment. The entry points above resolve the bundled dataset automatically; per-condition fine-tuning settings are stored in `runs/`.

The supplied checkpoints and default fine-tuning seed are **2026**. The paper's main table averages seeds 2026, 2027, and 2028, so the supplied single-seed results can differ slightly from those averages.
