import copy
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import main_mprgnetv2_single_scene as single
from main_ours import compute_metrics, inverse_pv_scale, measure_inference_time
from model.MPRGNetv2 import Model


METRICS = ('MAE', 'RMSE', 'WAPE', 'R2')


@torch.no_grad()
def evaluate_checkpoint(checkpoint, dataset, meta, args, cli):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = Model(SimpleNamespace(**saved['config'])).to(args.device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.mode = 'soft'
    model.set_stage('eval').eval()
    preds, truths = [], []
    for raw in single.loader(dataset, args, 16 if cli.smoke else 0):
        batch = single.move(raw, args.device)
        pred = model(batch['x'])['pred']
        if pred.shape != batch['y'].shape or not torch.isfinite(pred).all():
            raise RuntimeError('Invalid test prediction')
        preds.append(pred.cpu().numpy())
        truths.append(batch['y'].cpu().numpy())
    predictions = inverse_pv_scale(np.concatenate(preds), meta['pv_scaler'], args.target_dim)
    targets = inverse_pv_scale(np.concatenate(truths), meta['pv_scaler'], args.target_dim)
    scores = compute_metrics(predictions, targets)
    if not all(np.isfinite(scores[m]) for m in METRICS):
        raise RuntimeError('Nonfinite test metric')
    total = sum(p.numel() for p in model.parameters())
    unused = sum(p.numel() for name, p in model.named_parameters()
                 if name.startswith(('obs_error_head.', 'forecast_head.projection.')))
    timing_args = copy.deepcopy(args)
    timing_args.batch_size = cli.timing_batch_size
    timing_args.timing_warmup_batches = 5
    timing_args.timing_measure_batches = 30
    timing_loader = DataLoader(dataset, batch_size=cli.timing_batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    times = [measure_inference_time(model, timing_loader, torch.device(args.device), timing_args, False)
             for _ in range(cli.timing_repetitions)]
    return dict(metrics=scores, params_total=total, params_K=total / 1000,
                registered_unused_params=unused,
                timing=dict(ms_per_sample=float(np.mean([r['ms_per_sample'] for r in times])),
                    repeat_sample_std=float(np.std([r['ms_per_sample'] for r in times], ddof=1)) if len(times) > 1 else None,
                    repetitions=times),
                test_windows=16 if cli.smoke else len(dataset), checkpoint_sha256=single.sha(checkpoint))
