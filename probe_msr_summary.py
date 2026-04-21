from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_train_module(repo_root: str):
    path = os.path.join(repo_root, 'train-msr-flex.py')
    spec = importlib.util.spec_from_file_location('train_msr_flex', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Probe pooled state vs summary alignment on MSR checkpoints.')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--clf-epochs', default=20, type=int)
    parser.add_argument('--reg-epochs', default=15, type=int)
    parser.add_argument('--probe-lr', default=3e-3, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    return parser.parse_args()


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_probe(model_type: str, input_dim: int, output_dim: int) -> nn.Module:
    if model_type == 'linear':
        return nn.Linear(input_dim, output_dim)
    if model_type == 'mlp':
        hidden = max(64, min(256, input_dim * 2))
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )
    raise ValueError(f'Unsupported model_type: {model_type}')


def video_accuracy(prob: torch.Tensor, label: torch.Tensor, video_idx: torch.Tensor) -> float:
    video_prob: Dict[int, torch.Tensor] = {}
    video_label: Dict[int, int] = {}
    for p, y, vid in zip(prob, label, video_idx):
        key = int(vid.item())
        if key in video_prob:
            video_prob[key] = video_prob[key] + p
        else:
            video_prob[key] = p.clone()
            video_label[key] = int(y.item())
    if not video_prob:
        return 0.0
    correct = 0
    for key, summed in video_prob.items():
        pred = int(summed.argmax().item())
        correct += int(pred == video_label[key])
    return float(correct / len(video_prob))


def evaluate_classifier(model: nn.Module, features: torch.Tensor, labels: torch.Tensor, video_idx: torch.Tensor, batch_size: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            batch = features[start : start + batch_size].to(device)
            all_logits.append(model(batch).cpu())
    logits = torch.cat(all_logits, dim=0)
    loss = F.cross_entropy(logits, labels).item()
    pred = logits.argmax(dim=1)
    clip_acc = float((pred == labels).float().mean().item())
    prob = F.softmax(logits, dim=1)
    return {
        'loss': loss,
        'clip_acc': clip_acc,
        'video_acc': video_accuracy(prob, labels, video_idx),
    }


def train_classifier(
    features_train: torch.Tensor,
    labels_train: torch.Tensor,
    video_idx_train: torch.Tensor,
    features_val: torch.Tensor,
    labels_val: torch.Tensor,
    video_idx_val: torch.Tensor,
    model_type: str,
    num_classes: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, float]:
    model = build_probe(model_type, features_train.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(features_train, labels_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    best = {'best_val_video_acc': -1.0}
    for epoch in range(epochs):
        model.train()
        for batch_feature, batch_label in loader:
            batch_feature = batch_feature.to(device)
            batch_label = batch_label.to(device)
            logits = model(batch_feature)
            loss = F.cross_entropy(logits, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = evaluate_classifier(model, features_val, labels_val, video_idx_val, batch_size, device)
        if metrics['video_acc'] >= best['best_val_video_acc']:
            best = {
                'epoch': epoch,
                'best_val_video_acc': metrics['video_acc'],
                'best_val_clip_acc': metrics['clip_acc'],
                'best_val_loss': metrics['loss'],
                'train_clip_acc': evaluate_classifier(model, features_train, labels_train, video_idx_train, batch_size, device)['clip_acc'],
            }
    return best


def evaluate_regressor(model: nn.Module, features: torch.Tensor, target: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            batch = features[start : start + batch_size].to(device)
            pred.append(model(batch).cpu())
    pred = torch.cat(pred, dim=0)
    return float(F.mse_loss(pred, target).item())


def train_regressor(
    features_train: torch.Tensor,
    target_train: torch.Tensor,
    features_val: torch.Tensor,
    target_val: torch.Tensor,
    model_type: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, float]:
    model = build_probe(model_type, features_train.size(1), target_train.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dataset = TensorDataset(features_train, target_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    best = {'best_val_mse': float('inf')}
    for epoch in range(epochs):
        model.train()
        for batch_feature, batch_target in loader:
            batch_feature = batch_feature.to(device)
            batch_target = batch_target.to(device)
            pred = model(batch_feature)
            loss = F.mse_loss(pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_mse = evaluate_regressor(model, features_val, target_val, batch_size, device)
        if val_mse <= best['best_val_mse']:
            best = {
                'epoch': epoch,
                'best_val_mse': val_mse,
                'train_mse': evaluate_regressor(model, features_train, target_train, batch_size, device),
            }
    return best


def concat_rows(rows):
    return torch.cat(rows, dim=0) if rows else torch.empty(0)


def extract_features(model, loader: DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
    pooled_final = []
    summary_final = []
    target_final = []
    labels = []
    video_idx = []
    pooled_prev = []
    pooled_delta = []
    summary_prev = []
    summary_delta = []
    target_prev = []
    target_delta = []
    target_name = None
    pred_target_sqerr = 0.0
    pred_target_count = 0

    model.eval()
    with torch.no_grad():
        for clip, target, vid in loader:
            clip = clip.to(device, non_blocking=True)
            output = model(clip, return_aux=True, return_sequence=True)
            frame_states = output['frame_states']
            summary_tokens = output['summary_tokens']
            aux_target_tokens = output.get('aux_target_tokens', summary_tokens)
            target_name = output.get('aux_target_name', 'summary')
            pooled_final.append(frame_states[:, -1].cpu())
            summary_final.append(summary_tokens[:, -1].reshape(summary_tokens.size(0), -1).cpu())
            target_final.append(aux_target_tokens[:, -1].reshape(aux_target_tokens.size(0), -1).cpu())
            labels.append(target.clone())
            video_idx.append(vid.clone())

            pooled_prev.append(frame_states[:, :-1].reshape(-1, frame_states.size(-1)).cpu())
            pooled_delta.append((frame_states[:, 1:] - frame_states[:, :-1]).reshape(-1, frame_states.size(-1)).cpu())

            summary_flat = summary_tokens.reshape(summary_tokens.size(0), summary_tokens.size(1), -1)
            summary_prev.append(summary_flat[:, :-1].reshape(-1, summary_flat.size(-1)).cpu())
            summary_delta.append((summary_flat[:, 1:] - summary_flat[:, :-1]).reshape(-1, summary_flat.size(-1)).cpu())

            target_flat = aux_target_tokens.reshape(aux_target_tokens.size(0), aux_target_tokens.size(1), -1)
            target_prev.append(target_flat[:, :-1].reshape(-1, target_flat.size(-1)).cpu())
            target_delta.append((target_flat[:, 1:] - target_flat[:, :-1]).reshape(-1, target_flat.size(-1)).cpu())

            if 1 in model.summary_predictor.horizons:
                pred_delta = model.summary_predictor(summary_tokens).get(1)
                if pred_delta is not None:
                    pred_target_delta = aux_target_tokens[:, 1:] - aux_target_tokens[:, :-1]
                    sqerr = (pred_delta - pred_target_delta).pow(2).mean(dim=(-1, -2, -3))
                    pred_target_sqerr += float(sqerr.sum().item())
                    pred_target_count += int(sqerr.numel())

    return {
        'pooled_final': concat_rows(pooled_final).float(),
        'summary_final': concat_rows(summary_final).float(),
        'target_final': concat_rows(target_final).float(),
        'labels': concat_rows(labels).long(),
        'video_idx': concat_rows(video_idx).long(),
        'pooled_prev': concat_rows(pooled_prev).float(),
        'pooled_delta': concat_rows(pooled_delta).float(),
        'summary_prev': concat_rows(summary_prev).float(),
        'summary_delta': concat_rows(summary_delta).float(),
        'target_prev': concat_rows(target_prev).float(),
        'target_delta': concat_rows(target_delta).float(),
        'target_name': target_name,
        'pred_target_mse_h1': float(pred_target_sqerr / max(pred_target_count, 1)),
    }


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    return {
        'dim': int(x.size(1)),
        'mean_abs': float(x.abs().mean().item()),
        'std_mean': float(x.std(dim=0, unbiased=False).mean().item()),
        'std_min': float(x.std(dim=0, unbiased=False).min().item()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    train_module = load_train_module(repo_root)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint['config']
    device = torch.device(cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    train_dataset, val_dataset = train_module.build_datasets(cfg)
    num_classes = train_module.dataset_num_classes(train_dataset)
    model = train_module.build_model(cfg, num_classes=num_classes)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    if cfg['model']['name'] != 'MAMBA4D_CausalPSD' or not cfg['model']['psd']['enable']:
        raise ValueError('Probe requires a PSD-enabled causal checkpoint.')

    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, shuffle=False)
    val_loader = build_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    train_feat = extract_features(model, train_loader, device)
    val_feat = extract_features(model, val_loader, device)

    results = {
        'checkpoint': args.checkpoint,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'feature_stats': {
            'pooled_final': summarize_tensor(train_feat['pooled_final']),
            'summary_final': summarize_tensor(train_feat['summary_final']),
        },
        'predictability': {
            'checkpoint_pred_target': train_feat['target_name'],
            'pred_target_mse_h1_checkpoint_train': train_feat['pred_target_mse_h1'],
            'pred_target_mse_h1_checkpoint_val': val_feat['pred_target_mse_h1'],
        },
        'classification': {},
        'delta_regression': {},
    }
    if train_feat['target_name'] != 'summary':
        results['feature_stats'][train_feat['target_name']] = summarize_tensor(train_feat['target_final'])

    for repr_name in ['pooled_final', 'summary_final']:
        results['classification'][repr_name] = {}
        for probe_type in ['linear', 'mlp']:
            metrics = train_classifier(
                features_train=train_feat[repr_name],
                labels_train=train_feat['labels'],
                video_idx_train=train_feat['video_idx'],
                features_val=val_feat[repr_name],
                labels_val=val_feat['labels'],
                video_idx_val=val_feat['video_idx'],
                model_type=probe_type,
                num_classes=num_classes,
                batch_size=args.batch_size,
                epochs=args.clf_epochs,
                lr=args.probe_lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            results['classification'][repr_name][probe_type] = metrics

    regression_specs = [
        ('pooled_prev', 'pooled_delta', 'pooled_state'),
        ('summary_prev', 'summary_delta', 'summary'),
    ]
    if train_feat['target_name'] != 'summary':
        regression_specs.append(('target_prev', 'target_delta', train_feat['target_name']))

    for source_name, target_name, repr_name in regression_specs:
        results['delta_regression'][repr_name] = {}
        for probe_type in ['linear', 'mlp']:
            metrics = train_regressor(
                features_train=train_feat[source_name],
                target_train=train_feat[target_name],
                features_val=val_feat[source_name],
                target_val=val_feat[target_name],
                model_type=probe_type,
                batch_size=max(args.batch_size, 128),
                epochs=args.reg_epochs,
                lr=args.probe_lr,
                weight_decay=args.weight_decay,
                device=device,
            )
            results['delta_regression'][repr_name][probe_type] = metrics

    with open(output_dir / 'probe_summary_alignment.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
