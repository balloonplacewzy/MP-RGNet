import copy
import json
from types import SimpleNamespace
import torch
import main_mprgnetv2_single_scene as single
import main_mprgnetv2_tune_long as tuning


ORIGINAL_ASSESS = tuning.assess


def assess(scores, reference):
    result = ORIGINAL_ASSESS(scores, reference)
    ratios = result['forecast_ratios']
    weighted = {k:.25*v['MAE']+.75*v['RMSE'] for k,v in ratios.items()}
    result['score'] = .5*weighted['full'] + .5*max(weighted[f'block{i}'] for i in (1,2,3))
    return result


def configure_selection():


    tuning.GUARDS.update(forecast_ratio=1.02, reconstruction_ratio=1.05)
    tuning.assess = assess


def trial_args(args, name, settings):
    result = copy.deepcopy(args)
    result.select_message_bound = settings['bound']
    result.tuning_candidate = name
    result.tuning_hyperparameters = settings
    result.tuning_round = 2
    return result


def completed_record(directory):
    path = directory/'training.json'
    if path.is_file():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record.get('complete'):
            assert single.sha(record['checkpoint']) == record['checkpoint_sha256']
            return record


def cap_control(directory, name, checkpoint, bound, args, validation, scale, reference):
    existing = completed_record(directory)
    if existing:
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = SimpleNamespace(**saved['config'])
    config.actual_backbone_checkpoint = args.actual_backbone_checkpoint
    config.actual_backbone_sha256 = args.actual_backbone_sha256
    for k in ('validation_windows','train_windows','quality_windows','quality_validation_windows',
              'quality_epochs','graph_epochs','graph_windows'):
        setattr(config,k,getattr(args,k))
    config = trial_args(config,name,dict(bound=bound, parent_checkpoint=str(checkpoint), evaluation_only=True))
    model = tuning.load_model(checkpoint,config)
    model.mode = 'soft' if name.startswith('control_previous') else 'off'
    metrics = tuning.validation_scores(model,validation,config,scale)
    score = assess(metrics,reference)
    tuning.save_model(directory/'best.pt',model,config,saved['selected_epoch'],'graph_bound_control')
    record = dict(candidate=name, complete=True, control=True, validation=metrics, assessment=score,
                  parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=single.sha(checkpoint),
                  checkpoint=str(directory/'best.pt'),checkpoint_sha256=single.sha(directory/'best.pt'))
    single.save_json(directory/'training.json',record)
    return record
