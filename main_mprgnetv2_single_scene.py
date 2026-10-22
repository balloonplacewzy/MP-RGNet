import copy
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from Dataprovider.pv_quality_dataset import PVDataConfig, build_pv_quality_dataloaders
from model.MPRGNetv2 import Model, quality_loss
from model.newmodelAUXGRAPH_QSELECT import Model as InitialGraphModel
ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def digest(model, prefixes):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(prefixes):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checkpoint(path, model, epoch, stage, args):
    temp = path.with_suffix('.pt.tmp')
    torch.save(dict(state_dict=state(model), config=model.resolved_config, selected_epoch=epoch,
                    stage=stage, seed=args.seed, scenario=args.scenario_name,
                    initialization=args.initialization,
                    source_checkpoint=args.actual_backbone_checkpoint,
                    source_sha256=args.actual_backbone_sha256), temp)
    temp.replace(path)


def loader(dataset, args, count=0, shuffle=False):
    ids = list(range(len(dataset))) if count == 0 else np.unique(
        np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)).tolist()
    return DataLoader(Subset(dataset, ids), batch_size=args.batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=args.num_workers,
                      pin_memory=str(args.device).startswith('cuda'),
                      generator=torch.Generator().manual_seed(args.seed))


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_data(args):
    cfg = PVDataConfig(csv_path=args.data, num_sites=args.target_dim,
        seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size,
        train_ratio=args.split_ratio[0], val_ratio=args.split_ratio[1],
        scenario_name=args.scenario_name, model_type='ours', impute_method='zero',
        seed=args.data_seed, data_seed=args.data_seed, num_workers=args.num_workers,
        pin_memory=str(args.device).startswith('cuda'), drop_last=False,
        day_threshold=args.day_threshold, scale=args.scale, aux_layout=args.aux_layout,
        aux_delay_steps=args.aux_delay_steps, missing_pattern='random_point',
        block_len=args.block_len, append_obs_mask=True)
    train, val, test, meta = build_pv_quality_dataloaders(cfg)
    for dataset in (train.dataset, val.dataset, test.dataset):
        if type(dataset).__module__ != 'Dataprovider.pv_quality_dataset':
            raise RuntimeError('Unexpected dataset wrapper')
        if dataset.scenario_name != args.scenario_name:
            raise RuntimeError('Unexpected training scenario')
        assert dataset.indices[0] >= dataset.split_start
        assert dataset.indices[-1] + args.seq_len + args.pred_len <= dataset.split_end
    assert train.dataset.split_end == val.dataset.split_start
    assert val.dataset.split_end == test.dataset.split_start
    return train.dataset, val.dataset, test.dataset, meta


def relation_prior(training):
    if training.split_start != 0:
        raise ValueError('Relation prior must use the training split')
    x = training.y_clean[:training.split_end].astype(np.float64)
    x -= x.mean(1, keepdims=True)
    x -= x.mean(0, keepdims=True)
    x /= np.sqrt((x * x).sum(0, keepdims=True)).clip(1e-8)
    prior = np.clip(x.T @ x, -1, 1).astype(np.float32)
    np.fill_diagonal(prior, 0)
    return torch.from_numpy(prior)


def backbone_path(args):
    template = args.backbone_checkpoint_template
    if args.scenario_name == 'mixed_30' and args.seq_len == 96 and args.pred_len == 8:
        template = args.mixed30_backbone_checkpoint_template
    return ROOT / template.format(scene=args.scenario_name, seed=args.seed,
                                  seq_len=args.seq_len, pred_len=args.pred_len)


def initialize_backbone(args):
    source = backbone_path(args)
    if not source.is_file():
        raise FileNotFoundError('Original single-scene backbone required: ' + str(source))
    seed_all(args.seed)
    initial_graph = InitialGraphModel(args).state_graph.state_dict()
    seed_all(args.seed)
    model = Model(args).to(args.device)
    original = torch.load(source, map_location='cpu', weights_only=True)
    original = original.get('state_dict', original)
    migrated = {k.replace('restore_head.reliability_head.', 'restore_head.alpha_head.'): v
                for k, v in original.items() if k.startswith(('restore_head.', 'forecast_head.'))}
    current = model.state_dict()
    expected = {k for k in current if k.startswith(('restore_head.', 'forecast_head.'))}
    if set(migrated) != expected or any(migrated[k].shape != current[k].shape for k in expected):
        raise RuntimeError('Original backbone keys/shapes do not match MPRGNetv2')
    current.update(migrated)
    model.load_state_dict(current, strict=True)
    model.state_graph.load_state_dict(initial_graph, strict=True)
    assert all(torch.equal(model.state_dict()[k].cpu(), migrated[k]) for k in expected)
    args.actual_backbone_checkpoint = str(source)
    args.actual_backbone_sha256 = sha(source)
    return model


@torch.inference_mode()
def evaluate(model, batches, scale, args):
    model.eval()
    absolute, squared, count = 0., 0., 0
    for raw in batches:
        batch = move(raw, args.device)
        pred = model(batch['x'])['pred']
        if pred.shape != batch['y'].shape or not torch.isfinite(pred).all():
            raise RuntimeError('Invalid forecast')
        error = ((pred - batch['y']) * scale).double()
        absolute += error.abs().sum().item()
        squared += error.square().sum().item()
        count += error.numel()
    return dict(MAE=absolute / count, RMSE=float(np.sqrt(squared / count)), count=count)


@torch.inference_mode()
def quality_score(model, batches, args):
    model.eval()
    total, count = 0., 0
    for raw in batches:
        batch = move(raw, args.device)
        features = model.encode(batch['x'])
        predicted = model.quality_outputs(features)['state_error']
        target = (features['p_impute'] - batch['pv_clean']).abs()
        mask = (batch['obs_mask'] < .5) & (batch['day_mask'] > .5)
        error = (predicted[mask].double() - target[mask].double()).square()
        total += error.sum().item()
        count += error.numel()
    if count == 0:
        raise RuntimeError('No daytime missing values for q validation')
    return total / count


def report(directory, record, args, stage, row):
    record[stage].append(row)
    save_json(directory / 'training.json', record)
    print(json.dumps(dict(scene=args.scenario_name, seed=args.seed, stage=stage, **row)), flush=True)


def graph_warmup(model, training, validation, scale, args, directory, record):
    model.set_stage('graph')
    model.mode = 'off'
    fixed = ('restore_head.', 'forecast_head.', 'state_error_head.', 'obs_error_head.')
    before = digest(model, fixed)
    optimizer = torch.optim.AdamW(model.state_graph.parameters(), lr=args.warmup_lr,
                                  weight_decay=args.weight_decay)
    batches = loader(training, args, args.warmup_windows, True)
    val_batches = loader(validation, args, args.validation_windows)
    for epoch in range(1, args.warmup_epochs + 1):
        model.train()
        total = 0.
        for raw in batches:
            batch = move(raw, args.device)
            features = model.encode(batch['x'])
            with torch.no_grad():
                reference = ((model.forecast_head(features['p_impute']) - batch['y']) * scale).abs()
                history_before = ((features['p_impute'] - batch['pv_clean']) * scale).abs()
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_from_features(features)
            error = ((output['pred'] - batch['y']) * scale).abs()
            guard = (error.mean(1) - reference.mean(1)).clamp_min(0).mean() / scale.mean()
            history_after = ((output['p_graph'] - batch['pv_clean']) * scale).abs()
            history = (history_after - history_before).clamp_min(0).mean() / scale.mean()
            loss = error.mean() / scale.mean() + args.graph_guard_weight * guard + args.warmup_history_weight * history
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.state_graph.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        score = evaluate(model, val_batches, scale, args)['MAE']
        report(directory, record, args, 'warmup', dict(epoch=epoch, loss=total / len(batches), validation_MAE=score))
    assert before == digest(model, fixed), 'Warmup changed original backbone'
    checkpoint(directory / 'warmup.pt', model, args.warmup_epochs, 'warmup', args)
    record['warmup_selected_epoch'] = args.warmup_epochs
    record['warmup_selection'] = 'fixed final epoch'


def prediction_stage(model, training, validation, scale, args, directory, record, stage):
    model.set_stage('joint')
    model.mode = 'off'
    core = [p for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith('state_graph.')]
    graph = list(model.state_graph.parameters())
    optimizer = torch.optim.AdamW([{'params': core, 'lr': args.lr},
                                   {'params': graph, 'lr': args.joint_graph_lr}],
                                  weight_decay=args.weight_decay)
    epochs, patience, decay = args.epochs, args.patience, args.decay_patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=decay, min_lr=1e-6)
    batches = loader(training, args, args.train_windows, True)
    val_batches = loader(validation, args, args.validation_windows)
    best = evaluate(model, val_batches, scale, args)['MAE']
    best_epoch, best_state, stale = 0, state(model), 0
    record[stage + '_initial_validation_MAE'] = best
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total = 0.
        for raw in batches:
            batch = move(raw, args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch['x'])
            loss = ((output['pred'] - batch['y']) * scale).abs().mean() / scale.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core + graph, args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        score = evaluate(model, val_batches, scale, args)['MAE']
        scheduler.step(score)
        if score < best:
            best, best_epoch, best_state, stale = score, epoch, state(model), 0
            checkpoint(directory / (stage + '.pt'), model, epoch, stage, args)
        else:
            stale += 1
        report(directory, record, args, stage, dict(epoch=epoch, loss=total / len(batches),
            validation_MAE=score, best_MAE=best, best_epoch=best_epoch,
            lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter() - started))
        if stale >= patience:
            break
    model.load_state_dict(best_state, strict=True)
    checkpoint(directory / (stage + '.pt'), model, best_epoch, stage, args)
    record[stage + '_selected_epoch'] = best_epoch


def train_one(training, validation, meta, args, directory):
    model = initialize_backbone(args)
    model.set_relation_prior(relation_prior(training), split_name='train')
    scale = torch.as_tensor(meta['pv_scaler'].std, device=args.device, dtype=torch.float32).reshape(1, 1, -1)
    record = dict(initialization=args.initialization, source_checkpoint=args.actual_backbone_checkpoint,
                  source_sha256=args.actual_backbone_sha256,
                  training_scenes=[args.scenario_name], warmup=[], joint=[], quality=[], graph=[])
    graph_warmup(model, training, validation, scale, args, directory, record)
    warmup_state = state(model)
    del model
    seed_all(args.seed)
    model = Model(args).to(args.device)
    model.load_pretrained_core(warmup_state)
    del warmup_state
    prediction_stage(model, training, validation, scale, args, directory, record, 'joint')
    selected = state(model)
    del model
    seed_all(args.seed)
    model = Model(args).to(args.device)
    model.load_pretrained_core(selected)
    del selected
    model.mode = 'soft'
    model.set_stage('quality')
    fixed = ('restore_head.', 'forecast_head.', 'state_graph.', 'obs_error_head.')
    before = digest(model, fixed)
    optimizer = torch.optim.AdamW(model.state_error_head.parameters(), lr=args.quality_lr,
                                  weight_decay=args.weight_decay)
    batches = loader(training, args, args.quality_windows, True)
    val_batches = loader(validation, args, args.quality_validation_windows)
    best, best_epoch, best_head = float('inf'), 0, None
    for epoch in range(1, args.quality_epochs + 1):
        model.train()
        total = 0.
        for raw in batches:
            batch = move(raw, args.device)
            features = model.encode(batch['x'])
            optimizer.zero_grad(set_to_none=True)
            loss = quality_loss(features, model.quality_outputs(features), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.state_error_head.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        score = quality_score(model, val_batches, args)
        if score < best:
            best, best_epoch, best_head = score, epoch, state(model.state_error_head)
        report(directory, record, args, 'quality', dict(epoch=epoch, loss=total / len(batches),
            validation_error_MSE=score, best_epoch=best_epoch))
    model.state_error_head.load_state_dict(best_head, strict=True)
    assert before == digest(model, fixed), 'q training changed forecast parameters'
    checkpoint(directory / 'quality.pt', model, best_epoch, 'quality', args)
    record['quality_selected_epoch'] = best_epoch
    del optimizer, best_head

    seed_all(args.seed)
    model.set_stage('graph')
    base = copy.deepcopy(model).set_stage('eval').eval()
    base.mode = 'off'
    fixed = ('restore_head.', 'forecast_head.', 'state_error_head.', 'obs_error_head.')
    before = digest(model, fixed)
    optimizer = torch.optim.AdamW(model.state_graph.parameters(), lr=args.graph_lr,
                                  weight_decay=args.weight_decay)
    batches = loader(training, args, args.graph_windows, True)
    val_batches = loader(validation, args, args.validation_windows)
    best = evaluate(model, val_batches, scale, args)['MAE']
    best_epoch, best_state = 0, state(model)
    for epoch in range(1, args.graph_epochs + 1):
        model.train()
        total = 0.
        for raw in batches:
            batch = move(raw, args.device)
            features = model.encode(batch['x'])
            with torch.no_grad():
                reference = ((base.forward_from_features(features)['pred'] - batch['y']) * scale).abs()
            optimizer.zero_grad(set_to_none=True)
            error = ((model.forward_from_features(features)['pred'] - batch['y']) * scale).abs()
            guard = (error.mean(1) - reference.mean(1)).clamp_min(0).mean() / scale.mean()
            loss = error.mean() / scale.mean() + args.graph_guard_weight * guard
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.state_graph.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()
        score = evaluate(model, val_batches, scale, args)['MAE']
        if score < best:
            best, best_epoch, best_state = score, epoch, state(model)
        report(directory, record, args, 'graph', dict(epoch=epoch, loss=total / len(batches),
            validation_MAE=score, best_MAE=best, best_epoch=best_epoch))
    model.load_state_dict(best_state, strict=True)
    assert before == digest(model, fixed), 'Graph adaptation changed frozen parameters'
    model.set_stage('eval').eval()
    checkpoint(directory / 'best.pt', model, best_epoch, 'graph', args)
    record.update(graph_selected_epoch=best_epoch, frozen_parameters_verified=True,
                  validation=evaluate(model, val_batches, scale, args), training_complete=True)
    save_json(directory / 'training.json', record)
    return record
