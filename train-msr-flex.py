from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset

import utils
from datasets.msr import MSRAction3D
from scheduler import WarmupMultiStepLR
import models.msr as BaselineModels
from models.msr_causal_psd import MAMBA4D_CausalPSD


DEFAULT_CONFIG: Dict[str, Any] = {
    'seed': 0,
    'output_dir': None,
    'device': 'cuda',
    'data': {
        'path': None,
        'clip_len': 8,
        'frame_interval': 1,
        'num_points': 512,
        'train_subset_size': None,
        'val_subset_size': None,
    },
    'model': {
        'name': 'MAMBA4D',
        'radius': 0.7,
        'nsamples': 16,
        'spatial_stride': 32,
        'temporal_kernel_size': 3,
        'temporal_stride': 2,
        'emb_relu': False,
        'dim': 128,
        'mlp_dim': 256,
        'depth_mamba_inter': 2,
        'depth_mamba_intra': 2,
        'rms_norm': False,
        'drop_out_in_block': 0.0,
        'drop_path': 0.0,
        'intra': False,
        'strict_causal': False,
        'point_order': 'none',
        'psd': {
            'enable': False,
            'num_slots': 4,
            'slot_dim': 64,
            'summary_mode': 'pooled_mlp',
            'horizons': [1, 2, 4],
            'pred_target_type': 'summary_delta',
            'predictor_type': 'mlp',
            'predictor_hidden_dim': 128,
            'predictor_layers': 1,
            'predictor_heads': 4,
            'summary_hidden_dim': None,
            'task_target_proj_hidden_dim': None,
            'sem_head_enable': False,
            'sem_hidden_dim': None,
            'sem_loss_type': 'hard_label_ce',
            'sem_temperature': 1.0,
            'dropout': 0.0,
            'reg_type': 'var_cov',
            'reg_var_target': 1.0,
            'reg_cov_weight': 1.0,
            'reg_sig_floor': 0.25,
            'reg_sig_uniform_weight': 0.1,
        },
    },
    'loss': {
        'lambda_pred': 0.0,
        'lambda_reg': 0.0,
        'lambda_div': 0.0,
        'lambda_sem': 0.0,
    },
    'train': {
        'batch_size': 8,
        'epochs': 3,
        'workers': 2,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'lr_milestones': [2],
        'lr_gamma': 0.1,
        'lr_warmup_epochs': 0,
        'aux_warmup_enable': False,
        'aux_warmup_ratio': 0.2,
        'aux_warmup_epochs': None,
        'aux_pred_start_factor': 0.25,
        'aux_reg_start_factor': 0.0,
        'aux_div_start_factor': 0.0,
        'aux_sem_start_factor': 1.0,
        'print_freq': 10,
    },
    'eval': {
        'run_causality_check': False,
        'causality_num_samples': 4,
        'causality_tolerance': 1e-6,
    },
    'resume': None,
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Flexible MSR trainer for baseline and causal+PSD runs')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--output-dir', default=None, type=str)
    parser.add_argument('--data-path', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    with open(args.config, 'r') as f:
        user_cfg = yaml.safe_load(f) or {}
    deep_update(cfg, user_cfg)
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.data_path is not None:
        cfg['data']['path'] = args.data_path
    if args.resume is not None:
        cfg['resume'] = args.resume
    if cfg['data']['path'] is None:
        raise ValueError('data.path must be provided either in config or via --data-path')
    return cfg


def setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_subset(dataset, subset_size: int | None, seed: int):
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dataset))[:subset_size].tolist()
    return Subset(dataset, indices)


def dataset_num_classes(dataset) -> int:
    if hasattr(dataset, 'num_classes'):
        return dataset.num_classes
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'num_classes'):
        return dataset.dataset.num_classes
    raise AttributeError('num_classes not found')


def build_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg['data']
    train_dataset = MSRAction3D(
        root=data_cfg['path'],
        frames_per_clip=data_cfg['clip_len'],
        frame_interval=data_cfg['frame_interval'],
        num_points=data_cfg['num_points'],
        train=True,
    )
    val_dataset = MSRAction3D(
        root=data_cfg['path'],
        frames_per_clip=data_cfg['clip_len'],
        frame_interval=data_cfg['frame_interval'],
        num_points=data_cfg['num_points'],
        train=False,
    )
    train_dataset = maybe_subset(train_dataset, data_cfg.get('train_subset_size'), cfg['seed'])
    val_dataset = maybe_subset(val_dataset, data_cfg.get('val_subset_size'), cfg['seed'] + 1)
    return train_dataset, val_dataset


def build_loaders(cfg: Dict[str, Any], train_dataset, val_dataset):
    workers = cfg['train']['workers']
    loader_kwargs = dict(
        batch_size=cfg['train']['batch_size'],
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_model(cfg: Dict[str, Any], num_classes: int) -> nn.Module:
    model_cfg = cfg['model']
    name = model_cfg['name']
    if name == 'MAMBA4D':
        Model = getattr(BaselineModels, name)
        model = Model(
            radius=model_cfg['radius'],
            nsamples=model_cfg['nsamples'],
            spatial_stride=model_cfg['spatial_stride'],
            temporal_kernel_size=model_cfg['temporal_kernel_size'],
            temporal_stride=model_cfg['temporal_stride'],
            emb_relu=model_cfg['emb_relu'],
            dim=model_cfg['dim'],
            mlp_dim=model_cfg['mlp_dim'],
            num_classes=num_classes,
            depth_mamba_inter=model_cfg['depth_mamba_inter'],
            rms_norm=model_cfg['rms_norm'],
            drop_out_in_block=model_cfg['drop_out_in_block'],
            drop_path=model_cfg['drop_path'],
            depth_mamba_intra=model_cfg['depth_mamba_intra'],
            intra=model_cfg['intra'],
        )
    elif name == 'MAMBA4D_CausalPSD':
        psd_cfg = model_cfg['psd']
        model = MAMBA4D_CausalPSD(
            radius=model_cfg['radius'],
            nsamples=model_cfg['nsamples'],
            spatial_stride=model_cfg['spatial_stride'],
            temporal_kernel_size=model_cfg['temporal_kernel_size'],
            temporal_stride=model_cfg['temporal_stride'],
            emb_relu=model_cfg['emb_relu'],
            dim=model_cfg['dim'],
            mlp_dim=model_cfg['mlp_dim'],
            num_classes=num_classes,
            depth_mamba_inter=model_cfg['depth_mamba_inter'],
            rms_norm=model_cfg['rms_norm'],
            drop_out_in_block=model_cfg['drop_out_in_block'],
            drop_path=model_cfg['drop_path'],
            intra=model_cfg['intra'],
            strict_causal=model_cfg['strict_causal'],
            point_order=model_cfg.get('point_order', 'none'),
            psd_enable=psd_cfg['enable'],
            psd_num_slots=psd_cfg['num_slots'],
            psd_slot_dim=psd_cfg['slot_dim'],
            psd_summary_mode=psd_cfg.get('summary_mode', 'pooled_mlp'),
            psd_horizons=psd_cfg['horizons'],
            psd_pred_target_type=psd_cfg.get('pred_target_type', 'summary_delta'),
            psd_predictor_type=psd_cfg['predictor_type'],
            psd_predictor_hidden_dim=psd_cfg['predictor_hidden_dim'],
            psd_predictor_layers=psd_cfg['predictor_layers'],
            psd_predictor_heads=psd_cfg['predictor_heads'],
            psd_summary_hidden_dim=psd_cfg['summary_hidden_dim'],
            psd_task_target_proj_hidden_dim=psd_cfg.get('task_target_proj_hidden_dim'),
            psd_dropout=psd_cfg['dropout'],
            psd_reg_type=psd_cfg['reg_type'],
            psd_reg_var_target=psd_cfg['reg_var_target'],
            psd_reg_cov_weight=psd_cfg['reg_cov_weight'],
            psd_reg_sig_floor=psd_cfg['reg_sig_floor'],
            psd_reg_sig_uniform_weight=psd_cfg['reg_sig_uniform_weight'],
            psd_sem_head_enable=psd_cfg.get('sem_head_enable', False),
            psd_sem_hidden_dim=psd_cfg.get('sem_hidden_dim'),
            psd_sem_loss_type=psd_cfg.get('sem_loss_type', 'hard_label_ce'),
            psd_sem_temperature=psd_cfg.get('sem_temperature', 1.0),
        )
    else:
        raise ValueError(f'Unsupported model name: {name}')
    return model


def unpack_model_output(output):
    if isinstance(output, dict):
        return output['logits'], output.get('aux_losses', {}), output.get('aux_stats', {})
    return output, {}, {}


def tensor_or_zero(value, device):
    if value is None:
        return torch.zeros((), device=device)
    return value


def compute_aux_loss_weights(cfg: Dict[str, Any], epoch: int) -> Dict[str, float]:
    loss_cfg = cfg['loss']
    train_cfg = cfg['train']
    weights = {
        'pred': float(loss_cfg.get('lambda_pred', 0.0)),
        'reg': float(loss_cfg.get('lambda_reg', 0.0)),
        'div': float(loss_cfg.get('lambda_div', 0.0)),
        'sem': float(loss_cfg.get('lambda_sem', 0.0)),
    }
    if not train_cfg.get('aux_warmup_enable', False):
        return weights

    warmup_epochs = train_cfg.get('aux_warmup_epochs')
    if warmup_epochs is None:
        warmup_ratio = float(train_cfg.get('aux_warmup_ratio', 0.0))
        warmup_epochs = max(1, int(round(train_cfg['epochs'] * warmup_ratio))) if warmup_ratio > 0 else 0
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs <= 0:
        return weights

    progress = min(float(epoch + 1) / float(warmup_epochs), 1.0)

    def scaled(name: str, start_key: str) -> float:
        start = float(train_cfg.get(start_key, 1.0))
        return weights[name] * (start + (1.0 - start) * progress)

    return {
        'pred': scaled('pred', 'aux_pred_start_factor'),
        'reg': scaled('reg', 'aux_reg_start_factor'),
        'div': scaled('div', 'aux_div_start_factor'),
        'sem': scaled('sem', 'aux_sem_start_factor'),
    }


def train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, cfg):
    model.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = f'Epoch: [{epoch}]'
    aux_weights = compute_aux_loss_weights(cfg, epoch)

    for clip, target, _ in metric_logger.log_every(data_loader, cfg['train']['print_freq'], header):
        start_time = time.time()
        clip, target = clip.to(device), target.to(device)

        if cfg['model']['name'] == 'MAMBA4D_CausalPSD':
            output = model(clip, target=target, return_aux=True)
        else:
            output = model(clip)
        logits, aux_losses, aux_stats = unpack_model_output(output)
        loss_cls = criterion(logits, target)
        loss_pred = tensor_or_zero(aux_losses.get('pred'), logits.device)
        loss_reg = tensor_or_zero(aux_losses.get('reg'), logits.device)
        loss_div = tensor_or_zero(aux_losses.get('div'), logits.device)
        loss_sem = tensor_or_zero(aux_losses.get('sem'), logits.device)
        loss = (
            loss_cls
            + aux_weights['pred'] * loss_pred
            + aux_weights['reg'] * loss_reg
            + aux_weights['div'] * loss_div
            + aux_weights['sem'] * loss_sem
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        batch_size = clip.shape[0]
        metric_logger.update(
            loss=loss.item(),
            loss_cls=loss_cls.item(),
            loss_pred=float(loss_pred.item()),
            loss_reg=float(loss_reg.item()),
            loss_div=float(loss_div.item()),
            loss_sem=float(loss_sem.item()),
            lambda_pred_eff=aux_weights['pred'],
            lambda_reg_eff=aux_weights['reg'],
            lambda_div_eff=aux_weights['div'],
            lambda_sem_eff=aux_weights['sem'],
            lr=optimizer.param_groups[0]['lr'],
        )
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['clips/s'].update(batch_size / max(time.time() - start_time, 1e-6))
        for key, value in aux_stats.items():
            metric_logger.update(**{key: float(value.item() if torch.is_tensor(value) else value)})

    metric_logger.synchronize_between_processes()
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return train_stats


def evaluate(model, criterion, data_loader, device, print_freq):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter='  ')
    header = 'Eval:'
    video_prob = {}
    video_label = {}
    num_classes = dataset_num_classes(data_loader.dataset)

    with torch.no_grad():
        for clip, target, video_idx in metric_logger.log_every(data_loader, print_freq, header):
            clip = clip.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(clip)
            logits = output['logits'] if isinstance(output, dict) else output
            loss = criterion(logits, target)
            acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
            prob = F.softmax(logits, dim=1).cpu().numpy()
            batch_size = clip.shape[0]
            target_np = target.cpu().numpy()
            video_idx_np = video_idx.cpu().numpy()
            for i in range(batch_size):
                idx = int(video_idx_np[i])
                if idx in video_prob:
                    video_prob[idx] += prob[i]
                else:
                    video_prob[idx] = prob[i]
                    video_label[idx] = int(target_np[i])
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    video_pred = {k: int(np.argmax(v)) for k, v in video_prob.items()}
    pred_correct = [video_pred[k] == video_label[k] for k in video_pred]
    total_acc = float(np.mean(pred_correct)) if pred_correct else 0.0

    class_count = [0] * num_classes
    class_correct = [0] * num_classes
    for k, v in video_pred.items():
        label = video_label[k]
        class_count[label] += 1
        class_correct[label] += int(v == label)
    class_acc = [class_correct[c] / class_count[c] if class_count[c] > 0 else 0.0 for c in range(num_classes)]

    eval_stats = {
        'val_loss': metric_logger.loss.global_avg,
        'val_clip_acc1': metric_logger.acc1.global_avg,
        'val_clip_acc5': metric_logger.acc5.global_avg,
        'val_video_acc1': total_acc,
        'val_class_acc_mean': float(np.mean(class_acc)),
    }
    logging.info('Eval stats: %s', json.dumps(eval_stats, sort_keys=True))
    return eval_stats


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def save_history(history: Iterable[Dict[str, Any]], output_dir: str) -> None:
    history = list(history)
    save_json(os.path.join(output_dir, 'history.json'), history)
    if not history:
        return
    keys = sorted({k for row in history for k in row.keys()})
    with open(os.path.join(output_dir, 'history.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_history(history: Iterable[Dict[str, Any]], output_dir: str) -> None:
    history = list(history)
    if not history:
        return
    epochs = [row['epoch'] for row in history]

    def series(key):
        vals = []
        for row in history:
            vals.append(row.get(key))
        return vals

    plt.figure(figsize=(8, 5))
    for key in ['loss', 'loss_cls', 'val_loss']:
        vals = series(key)
        if any(v is not None for v in vals):
            plt.plot(epochs, vals, marker='o', label=key)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for key in ['acc1', 'val_clip_acc1', 'val_video_acc1']:
        vals = series(key)
        if any(v is not None for v in vals):
            plt.plot(epochs, vals, marker='o', label=key)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=200)
    plt.close()

    aux_keys = [
        'loss_pred',
        'loss_reg',
        'loss_div',
        'loss_sem',
        'summary_std',
        'summary_std_min',
        'slot_corr',
        'slot_max_corr',
        'task_target_std',
        'task_target_delta_abs_mean',
    ]
    if any(any(row.get(k) is not None for k in aux_keys) for row in history):
        plt.figure(figsize=(8, 5))
        for key in aux_keys:
            vals = series(key)
            if any(v is not None for v in vals):
                plt.plot(epochs, vals, marker='o', label=key)
        plt.xlabel('epoch')
        plt.ylabel('aux metric')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aux_curves.png'), dpi=200)
        plt.close()


def save_environment_snapshot(output_dir: str, cfg: Dict[str, Any]) -> None:
    env = {
        'python': sys.executable,
        'python_version': sys.version,
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'config': cfg,
    }
    save_json(os.path.join(output_dir, 'env.json'), env)
    with open(os.path.join(output_dir, 'resolved_config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_causality_sanity_check(model, dataset, device, num_samples: int = 4, tolerance: float = 1e-6):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    diffs = []
    checked = 0
    model.eval()
    with torch.no_grad():
        for clip, _, _ in loader:
            clip = clip.to(device)
            base = model(clip, return_sequence=True)
            if 'logits_seq' not in base or 'frame_indices' not in base:
                break
            logits_seq = base['logits_seq']
            steps = logits_seq.size(1)
            if steps < 2:
                continue
            step = min(steps - 2, max(0, steps // 2))
            raw_anchor = int(base['frame_indices'][step])
            if raw_anchor + 1 >= clip.size(1):
                continue
            perturbed = clip.clone()
            future = perturbed[:, raw_anchor + 1 :]
            scale = future.std().clamp_min(1e-6)
            perturbed[:, raw_anchor + 1 :] = torch.randn_like(future) * scale
            alt = model(perturbed, return_sequence=True)
            diff = (base['logits_seq'][:, step] - alt['logits_seq'][:, step]).abs().max().item()
            diffs.append(diff)
            checked += 1
            if checked >= num_samples:
                break
    report = {
        'checked_samples': checked,
        'max_abs_logit_diff': float(max(diffs) if diffs else 0.0),
        'mean_abs_logit_diff': float(np.mean(diffs) if diffs else 0.0),
        'tolerance': tolerance,
        'passed': bool((max(diffs) if diffs else 0.0) <= tolerance),
    }
    return report


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    if cfg['output_dir'] is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        cfg['output_dir'] = os.path.join('outputs', Path(args.config).stem + '-' + timestamp)
    setup_logging(cfg['output_dir'])
    save_environment_snapshot(cfg['output_dir'], cfg)
    set_seed(cfg['seed'])

    logging.info('Config loaded from %s', args.config)
    logging.info('Resolved config: %s', json.dumps(cfg, sort_keys=False))

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_loaders(cfg, train_dataset, val_dataset)
    logging.info('Train samples: %d | Val samples: %d', len(train_dataset), len(val_dataset))

    num_classes = dataset_num_classes(train_dataset)
    model = build_model(cfg, num_classes=num_classes).to(device)
    logging.info('Model params: %d', sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )
    warmup_iters = cfg['train']['lr_warmup_epochs'] * max(len(train_loader), 1)
    lr_milestones = [max(len(train_loader), 1) * m for m in cfg['train']['lr_milestones']]
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        milestones=lr_milestones,
        gamma=cfg['train']['lr_gamma'],
        warmup_iters=warmup_iters,
        warmup_factor=1e-5,
    )

    start_epoch = 0
    best_metric = -float('inf')
    history = []

    if cfg.get('resume'):
        checkpoint = torch.load(cfg['resume'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint.get('best_metric', best_metric)
        history = checkpoint.get('history', history)
        logging.info('Resumed from %s at epoch %d', cfg['resume'], start_epoch)

    start_time = time.time()
    for epoch in range(start_epoch, cfg['train']['epochs']):
        train_stats = train_one_epoch(model, criterion, optimizer, lr_scheduler, train_loader, device, epoch, cfg)
        eval_stats = evaluate(model, criterion, val_loader, device, cfg['train']['print_freq'])
        row = {'epoch': epoch, **train_stats, **eval_stats}
        history.append(row)
        save_history(history, cfg['output_dir'])
        plot_history(history, cfg['output_dir'])
        logging.info('Epoch summary: %s', json.dumps(row, sort_keys=True))

        metric = eval_stats['val_video_acc1']
        if metric >= best_metric:
            best_metric = metric
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
                'history': history,
                'config': cfg,
            }
            torch.save(checkpoint, os.path.join(cfg['output_dir'], 'checkpoint_best.pth'))
            logging.info('Saved new best checkpoint with val_video_acc1=%.6f', best_metric)

    total_time = time.time() - start_time
    best_row = max(history, key=lambda row: row.get('val_video_acc1', -float('inf'))) if history else None
    final_row = history[-1] if history else {}
    summary = {
        'best_val_video_acc1': best_metric,
        'best_epoch': best_row.get('epoch') if best_row else None,
        'final_val_video_acc1': final_row.get('val_video_acc1'),
        'final_val_clip_acc1': final_row.get('val_clip_acc1'),
        'total_time_sec': total_time,
        'epochs': cfg['train']['epochs'],
    }

    if cfg['model']['name'] == 'MAMBA4D_CausalPSD' and cfg['eval']['run_causality_check']:
        best_path = os.path.join(cfg['output_dir'], 'checkpoint_best.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            model.to(device)
        causality = run_causality_sanity_check(
            model,
            val_dataset,
            device,
            num_samples=cfg['eval']['causality_num_samples'],
            tolerance=cfg['eval']['causality_tolerance'],
        )
        summary['causality'] = causality
        save_json(os.path.join(cfg['output_dir'], 'causality_report.json'), causality)
        logging.info('Causality check: %s', json.dumps(causality, sort_keys=True))

    save_json(os.path.join(cfg['output_dir'], 'summary.json'), summary)
    logging.info('Run complete: %s', json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
