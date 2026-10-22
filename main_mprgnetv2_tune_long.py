import copy
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
import main_mprgnetv2_single_scene as single
from main_ours import weighted_impute_loss
from model.MPRGNetv2 import Model, quality_loss


CANDIDATES = {
    'low_lr': dict(lr=1e-5, graph_lr=3e-5, restoration_weight=0.0, freeze_restore=False),
    'restore_05': dict(lr=1e-5, graph_lr=3e-5, restoration_weight=0.5, freeze_restore=False),
    'restore_10': dict(lr=1e-5, graph_lr=3e-5, restoration_weight=1.0, freeze_restore=False),
    'freeze_restore': dict(lr=1e-5, graph_lr=3e-5, restoration_weight=0.0, freeze_restore=True),
}


GUARDS = dict(forecast_ratio=1.02, reconstruction_ratio=1.10)


def load_model(path, args=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    cfg = SimpleNamespace(**saved['config']) if args is None else copy.deepcopy(args)
    model = Model(cfg).to(cfg.device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.set_stage('eval').eval()
    model.mode = 'off'
    return model


def save_model(path, model, args, epoch, stage):
    model.resolved_config.update(vars(args))
    model.resolved_config.update(jointq_scope=model.state_graph.quality_scope,
                                jointq_strength=model.state_graph.quality_strength)
    single.checkpoint(path, model, epoch, stage, args)


@torch.inference_mode()
def validation_scores(model, dataset, args, scale):

    model.eval()
    boundaries = np.linspace(dataset.split_start, dataset.split_end, 4, dtype=int).tolist()
    meters = {k: [0.0, 0.0, 0] for k in ('full', 'block1', 'block2', 'block3')}
    reconstruction = [0.0, 0.0, 0]
    for raw in single.loader(dataset, args, args.validation_windows):
        batch = single.move(raw, args.device)
        out = model(batch['x'])
        error = ((out['pred'] - batch['y']) * scale).double()
        if not torch.isfinite(error).all():
            raise RuntimeError('Nonfinite validation predictions')
        indices = batch['index'].reshape(-1)
        masks = {'full': torch.ones_like(indices, dtype=torch.bool)}
        for i in range(3):
            masks[f'block{i+1}'] = ((indices >= boundaries[i]) &
                (indices + args.seq_len + args.pred_len <= boundaries[i+1]))
        for name, mask in masks.items():
            values = error[mask]
            meters[name][0] += values.abs().sum().item()
            meters[name][1] += values.square().sum().item()
            meters[name][2] += values.numel()
        missing = (batch['obs_mask'] < .5) & (batch['day_mask'] > .5)
        reconstruction[0] += (((out['p_impute'] - batch['pv_clean']) * scale)[missing]).double().abs().sum().item()
        reconstruction[1] += (((out['p_graph'] - batch['pv_clean']) * scale)[missing]).double().abs().sum().item()
        reconstruction[2] += missing.sum().item()
    if any(v[2] == 0 for v in meters.values()) or reconstruction[2] == 0:
        raise RuntimeError('Empty validation block or reconstruction mask')
    return dict(forecast={k: dict(MAE=a/n, RMSE=float(np.sqrt(s/n)), count=n)
                             for k, (a, s, n) in meters.items()},
                reconstruction_MAE=reconstruction[0]/reconstruction[2],
                graph_reconstruction_MAE=reconstruction[1]/reconstruction[2],
                reconstruction_count=reconstruction[2], block_boundaries=boundaries)


def assess(scores, reference):
    ratios = {name: {m: scores['forecast'][name][m]/max(reference['forecast'][name][m], 1e-12)
                     for m in ('MAE', 'RMSE')} for name in scores['forecast']}
    recon = {k: scores[k]/max(reference[k], 1e-12)
             for k in ('reconstruction_MAE', 'graph_reconstruction_MAE')}
    eligible = (max(v for group in ratios.values() for v in group.values()) <= GUARDS['forecast_ratio']
                and max(recon.values()) <= GUARDS['reconstruction_ratio'])

    full = float(np.mean(list(ratios['full'].values())))
    worst = max(float(np.mean(list(ratios[f'block{i}'].values()))) for i in range(1, 4))
    return dict(eligible=eligible, score=.5*full + .5*worst,
                forecast_ratios=ratios, reconstruction_ratios=recon)


def prepare_reference(directory, warmup, args, validation, scale):

    model = load_model(warmup, args)
    with torch.no_grad():
        model.state_graph.message_net[-1].weight.zero_()
        model.state_graph.message_net[-1].bias.zero_()
    metrics = validation_scores(model, validation, args, scale)
    save_model(directory / 'reference.pt', model, args, 0, 'original_core_identity_graph')
    return metrics


def train_candidate(cli, name, directory, warmup, reference_path, reference, training, validation, args, scale):
    record_path = directory / 'training.json'
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record.get('complete'):
            assert single.sha(directory / 'best.pt') == record['checkpoint_sha256']
            return record
    directory.mkdir(parents=True, exist_ok=True)
    config = CANDIDATES[name]
    single.seed_all(args.seed)
    args = copy.deepcopy(args)
    args.tuning_candidate = name
    args.tuning_hyperparameters = config
    model = load_model(warmup, args)
    fallback = load_model(reference_path, args)
    save_model(directory / 'best.pt', fallback, args, 0, 'reference_fallback')
    del fallback
    best_metrics, best_assessment, selected_epoch = reference, assess(reference, reference), 0
    initial = validation_scores(model, validation, args, scale)
    initial_assessment = assess(initial, reference)
    if initial_assessment['eligible'] and initial_assessment['score'] < best_assessment['score']:
        best_metrics, best_assessment = initial, initial_assessment
        save_model(directory / 'best.pt', model, args, 0, 'warmup')
    model.set_stage('joint')
    if config['freeze_restore']:
        model.restore_head.requires_grad_(False)
    frozen_before = single.digest(model, ('restore_head.',)) if config['freeze_restore'] else None
    core = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('state_graph.')]
    graph = list(model.state_graph.parameters())
    optimizer = torch.optim.AdamW([dict(params=core, lr=config['lr']),
                                   dict(params=graph, lr=config['graph_lr'])], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5, min_lr=1e-6)
    record = dict(candidate=name, hyperparameters=config, initial_validation=initial,
                  initial_assessment=initial_assessment, epochs=[], complete=False)
    loader = single.loader(training, args, args.train_windows, True)
    stale = 0
    for epoch in range(1, (1 if cli.smoke else cli.epochs) + 1):
        started = time.perf_counter()
        model.train()
        if config['freeze_restore']:
            model.restore_head.eval()
        losses = np.zeros(3)
        for raw in loader:
            batch = single.move(raw, args.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch['x'])
            forecast_loss = ((out['pred']-batch['y'])*scale).abs().mean()/scale.mean()
            restore_loss = weighted_impute_loss(out['p_impute'], batch, args)
            loss = forecast_loss + config['restoration_weight']*restore_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core + graph, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            losses += [loss.item(), forecast_loss.item(), restore_loss.item()]
        metrics = validation_scores(model, validation, args, scale)
        assessment = assess(metrics, reference)
        scheduler.step(assessment['score'])
        improved = assessment['eligible'] and assessment['score'] < best_assessment['score'] - 1e-6
        if improved:
            best_metrics, best_assessment, selected_epoch, stale = metrics, assessment, epoch, 0
            save_model(directory / 'best.pt', model, args, epoch, 'tuned_joint')
        else:
            stale += 1
        row = dict(epoch=epoch, loss=dict(zip(('total', 'forecast', 'restoration'), (losses/len(loader)).tolist())),
                   validation=metrics, assessment=assessment, selected_epoch=selected_epoch,
                   lr=[g['lr'] for g in optimizer.param_groups], seconds=time.perf_counter()-started)
        record['epochs'].append(row)
        single.save_json(record_path, record)
        print(json.dumps(dict(candidate=name, scene=args.scenario_name, horizon=args.pred_len,
                              epoch=epoch, eligible=assessment['eligible'], score=assessment['score'],
                              validation_MAE=metrics['forecast']['full']['MAE'], selected_epoch=selected_epoch)), flush=True)
        if epoch >= 4 and stale >= 4:
            break
    if frozen_before is not None:
        assert frozen_before == single.digest(model, ('restore_head.',)), 'Frozen reconstruction changed'
    record.update(complete=True, selected_epoch=selected_epoch, validation=best_metrics,
                  assessment=best_assessment, frozen_restore_verified=frozen_before is not None,
                  checkpoint=str(directory/'best.pt'), checkpoint_sha256=single.sha(directory/'best.pt'))
    single.save_json(record_path, record)
    return record


def calibrate_and_adapt(cli, directory, selected, reference, training, validation, args, scale):
    path = directory / 'final_training.json'
    if path.is_file():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('complete'):
            assert single.sha(directory/'best.pt') == record['checkpoint_sha256']
            return record
    training_seed = args.seed
    single.seed_all(training_seed)
    model = load_model(selected['checkpoint'])
    args = SimpleNamespace(**model.resolved_config)
    assert args.seed == training_seed, 'Calibration checkpoint and requested seed differ'
    args.actual_backbone_checkpoint = torch.load(selected['checkpoint'], map_location='cpu', weights_only=True)['source_checkpoint']
    args.actual_backbone_sha256 = single.sha(args.actual_backbone_checkpoint)
    best_metrics = validation_scores(model, validation, args, scale)
    best_assessment = assess(best_metrics, reference)
    assert best_assessment['eligible']
    save_model(directory/'best.pt', model, args, 0, 'tuned_joint_before_calibration')
    record = dict(selected_candidate=selected['candidate'], quality=[], graph=[], quality_modes=[], complete=False)
    model.mode = 'soft'
    model.set_stage('quality')
    frozen_prefixes = ('restore_head.', 'forecast_head.', 'state_graph.', 'obs_error_head.')
    frozen = single.digest(model, frozen_prefixes)
    optimizer = torch.optim.AdamW(model.state_error_head.parameters(), lr=args.quality_lr, weight_decay=args.weight_decay)
    loader = single.loader(training, args, args.quality_windows, True)
    val_loader = single.loader(validation, args, args.quality_validation_windows)
    best_quality, quality_epoch, quality_state = float('inf'), 0, None
    for epoch in range(1, args.quality_epochs+1):
        model.train()
        for raw in loader:
            batch = single.move(raw, args.device)
            features = model.encode(batch['x'])
            optimizer.zero_grad(set_to_none=True)
            loss = quality_loss(features, model.quality_outputs(features), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.state_error_head.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
        score = single.quality_score(model, val_loader, args)
        if score < best_quality:
            best_quality, quality_epoch, quality_state = score, epoch, single.state(model.state_error_head)
        record['quality'].append(dict(epoch=epoch, validation_error_MSE=score))
        single.save_json(path, record)
    model.state_error_head.load_state_dict(quality_state)
    assert frozen == single.digest(model, frozen_prefixes)
    model.set_stage('eval').eval()

    mode_best, mode_score = ('longgap', .5), float('inf')
    for scope, strength in [('longgap', .5), ('all', .1), ('all', .5)]:
        model.state_graph.quality_scope, model.state_graph.quality_strength = scope, strength
        metrics = validation_scores(model, validation, args, scale)
        assessment = assess(metrics, reference)
        record['quality_modes'].append(dict(scope=scope, strength=strength, validation=metrics, assessment=assessment))
        if assessment['eligible'] and assessment['score'] < mode_score:
            mode_best, mode_score = (scope, strength), assessment['score']
        if assessment['eligible'] and assessment['score'] <= best_assessment['score'] + 1e-10:
            best_metrics, best_assessment = metrics, assessment
            save_model(directory/'best.pt', model, args, quality_epoch, 'tuned_quality')
    model.state_graph.quality_scope, model.state_graph.quality_strength = mode_best
    model.set_stage('graph')
    frozen_prefixes = ('restore_head.', 'forecast_head.', 'state_error_head.', 'obs_error_head.')
    frozen = single.digest(model, frozen_prefixes)
    optimizer = torch.optim.AdamW(model.state_graph.parameters(), lr=1e-5, weight_decay=args.weight_decay)
    loader = single.loader(training, args, args.graph_windows, True)
    for epoch in range(1, args.graph_epochs+1):
        model.train()
        for raw in loader:
            batch = single.move(raw, args.device)
            optimizer.zero_grad(set_to_none=True)
            features = model.encode(batch['x'])
            out = model.forward_from_features(features)
            with torch.no_grad():
                baseline_error = ((model.forecast_head(features['p_impute'])-batch['y'])*scale).abs()
            error = ((out['pred']-batch['y'])*scale).abs()
            guard = (error.mean(1)-baseline_error.mean(1)).clamp_min(0).mean()/scale.mean()
            history = weighted_impute_loss(out['p_graph'], batch, args)
            loss = error.mean()/scale.mean() + .5*guard + .5*history
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.state_graph.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
        metrics = validation_scores(model, validation, args, scale)
        assessment = assess(metrics, reference)
        if assessment['eligible'] and assessment['score'] < best_assessment['score'] - 1e-6:
            best_metrics, best_assessment = metrics, assessment
            save_model(directory/'best.pt', model, args, epoch, 'tuned_graph')
        record['graph'].append(dict(epoch=epoch, validation=metrics, assessment=assessment))
        single.save_json(path, record)
    assert frozen == single.digest(model, frozen_prefixes)

    verified = load_model(directory/'best.pt')
    verified.mode = 'soft'
    metrics = validation_scores(verified, validation, args, scale)
    assert abs(metrics['forecast']['full']['MAE']-best_metrics['forecast']['full']['MAE']) < 1e-6
    assert assess(metrics, reference)['eligible']
    record.update(complete=True, quality_selected_epoch=quality_epoch, validation=metrics,
                  assessment=assess(metrics, reference), checkpoint=str(directory/'best.pt'),
                  checkpoint_sha256=single.sha(directory/'best.pt'), frozen_parameters_verified=True)
    single.save_json(path, record)
    return record
