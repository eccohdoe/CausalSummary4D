import os
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from anti_collapse import AntiCollapseRegularizer, slot_diversity_loss
from causal_p4d import CausalP4DConv
from mamba import MixerModel
from summary_extractor import SummaryExtractor
from summary_predictor import MultiHorizonSummaryPredictor


class MAMBA4D_CausalPSD(nn.Module):
    """Strict-causal backbone with training-only predictive summary dynamics."""

    def __init__(
        self,
        radius: float,
        nsamples: int,
        spatial_stride: int,
        temporal_kernel_size: int,
        temporal_stride: int,
        emb_relu: bool,
        dim: int,
        mlp_dim: int,
        num_classes: int,
        depth_mamba_inter: int,
        rms_norm: bool,
        drop_out_in_block: float,
        drop_path: float,
        intra: bool = False,
        strict_causal: bool = True,
        point_order: str = 'none',
        psd_enable: bool = False,
        psd_num_slots: int = 4,
        psd_slot_dim: int = 64,
        psd_summary_mode: str = 'pooled_mlp',
        psd_horizons: Iterable[int] = (1, 2, 4),
        psd_pred_target_type: str = 'summary_delta',
        psd_predictor_type: str = 'mlp',
        psd_predictor_hidden_dim: int = 128,
        psd_predictor_layers: int = 1,
        psd_predictor_heads: int = 4,
        psd_summary_hidden_dim: Optional[int] = None,
        psd_task_target_proj_hidden_dim: Optional[int] = None,
        psd_dropout: float = 0.0,
        psd_reg_type: str = 'var_cov',
        psd_reg_var_target: float = 1.0,
        psd_reg_cov_weight: float = 1.0,
        psd_reg_sig_floor: float = 0.25,
        psd_reg_sig_uniform_weight: float = 0.1,
        psd_sem_head_enable: bool = False,
        psd_sem_hidden_dim: Optional[int] = None,
        psd_sem_loss_type: str = 'hard_label_ce',
        psd_sem_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if intra:
            raise NotImplementedError('MAMBA4D_CausalPSD minimal version currently supports intra=False only.')

        self.strict_causal = strict_causal
        self.point_order = point_order
        self.psd_enable = psd_enable
        self.dim = dim
        self.psd_num_slots = psd_num_slots
        self.psd_slot_dim = psd_slot_dim
        self.psd_sem_head_enable = psd_sem_head_enable
        self.psd_pred_target_type = psd_pred_target_type
        self.psd_sem_loss_type = psd_sem_loss_type
        self.psd_sem_temperature = max(float(psd_sem_temperature), 1e-4)

        self.tube_embedding = CausalP4DConv(
            in_planes=0,
            mlp_planes=[dim],
            mlp_batch_norm=[False],
            mlp_activation=[False],
            spatial_kernel_size=[radius, nsamples],
            spatial_stride=spatial_stride,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            operator='+',
            spatial_pooling='max',
            temporal_pooling='mean',
        )
        self.pos_embedding = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.emb_relu = nn.ReLU() if emb_relu else nn.Identity()
        self.mamba_blocks = MixerModel(
            d_model=dim,
            n_layer=depth_mamba_inter,
            rms_norm=rms_norm,
            drop_out_in_block=drop_out_in_block,
            drop_path=drop_path,
        )
        self.frame_norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes),
        )

        if self.psd_enable:
            if self.psd_pred_target_type not in {'summary_delta', 'pooled_proj_delta'}:
                raise ValueError(f'Unsupported psd_pred_target_type: {self.psd_pred_target_type}')

            self.summary_extractor = SummaryExtractor(
                input_dim=dim,
                num_slots=psd_num_slots,
                slot_dim=psd_slot_dim,
                mode=psd_summary_mode,
                hidden_dim=psd_summary_hidden_dim,
                dropout=psd_dropout,
            )
            self.summary_predictor = MultiHorizonSummaryPredictor(
                slot_dim=psd_slot_dim,
                horizons=psd_horizons,
                predictor_type=psd_predictor_type,
                hidden_dim=psd_predictor_hidden_dim,
                num_layers=psd_predictor_layers,
                num_heads=psd_predictor_heads,
                dropout=psd_dropout,
            )
            self.anti_collapse = AntiCollapseRegularizer(
                reg_type=psd_reg_type,
                var_target=psd_reg_var_target,
                cov_weight=psd_reg_cov_weight,
                sig_floor=psd_reg_sig_floor,
                sig_uniform_weight=psd_reg_sig_uniform_weight,
            )
            self.pred_loss_fn = nn.SmoothL1Loss()

            if self.psd_pred_target_type == 'pooled_proj_delta':
                proj_hidden_dim = psd_task_target_proj_hidden_dim or max(dim, psd_num_slots * psd_slot_dim)
                self.task_target_proj = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, proj_hidden_dim),
                    nn.GELU(),
                    nn.Linear(proj_hidden_dim, psd_num_slots * psd_slot_dim),
                )
                self.task_target_norm = nn.LayerNorm(psd_slot_dim)

            if self.psd_sem_head_enable:
                sem_hidden_dim = psd_sem_hidden_dim or max(psd_slot_dim, mlp_dim // 2)
                self.summary_sem_head = nn.Sequential(
                    nn.LayerNorm(psd_slot_dim),
                    nn.Linear(psd_slot_dim, sem_hidden_dim),
                    nn.GELU(),
                    nn.Linear(sem_hidden_dim, num_classes),
                )
                self.sem_loss_fn = nn.CrossEntropyLoss()

    def _order_points(self, xyzs: torch.Tensor, features: torch.Tensor):
        if self.point_order == 'none':
            return xyzs, features
        if self.point_order == 'x':
            key = xyzs[..., 0]
        elif self.point_order == 'xyz_sum':
            key = xyzs[..., 0] + 1e-2 * xyzs[..., 1] + 1e-4 * xyzs[..., 2]
        else:
            raise ValueError(f'Unsupported point_order: {self.point_order}')
        idx = torch.argsort(key, dim=2)
        xyzs = torch.gather(xyzs, 2, idx.unsqueeze(-1).expand(-1, -1, -1, xyzs.size(-1)))
        features = torch.gather(features, 2, idx.unsqueeze(-1).expand(-1, -1, -1, features.size(-1)))
        return xyzs, features

    def _build_prediction_target(self, frame_states: torch.Tensor, summary_tokens: torch.Tensor):
        if self.psd_pred_target_type == 'summary_delta':
            return summary_tokens, 'summary'
        target_tokens = self.task_target_proj(frame_states)
        target_tokens = target_tokens.view(
            frame_states.size(0),
            frame_states.size(1),
            self.psd_num_slots,
            self.psd_slot_dim,
        )
        target_tokens = self.task_target_norm(target_tokens)
        return target_tokens, 'projected_pooled_target'

    def _soft_semantic_alignment_loss(
        self,
        summary_logits: torch.Tensor,
        main_logits: torch.Tensor,
    ) -> torch.Tensor:
        temp = self.psd_sem_temperature
        teacher_prob = F.softmax(main_logits.detach() / temp, dim=-1)
        student_log_prob = F.log_softmax(summary_logits / temp, dim=-1)
        return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temp * temp)

    def _build_auxiliary(
        self,
        frame_states: torch.Tensor,
        token_states: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        main_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        summary_tokens = self.summary_extractor(frame_states, token_states=token_states)
        pred_target_tokens, pred_target_name = self._build_prediction_target(frame_states, summary_tokens)
        pred_dict = self.summary_predictor(summary_tokens)
        zero = summary_tokens.new_zeros(())
        pred_losses = []
        pred_stats: Dict[str, torch.Tensor] = {}
        for horizon, pred_delta in pred_dict.items():
            target_delta = pred_target_tokens[:, horizon:] - pred_target_tokens[:, :-horizon]
            loss_h = self.pred_loss_fn(pred_delta, target_delta)
            pred_losses.append(loss_h)
            pred_stats[f'pred_h{horizon}'] = loss_h.detach()
        pred_loss = torch.stack(pred_losses).mean() if pred_losses else zero
        reg_loss, reg_stats = self.anti_collapse(summary_tokens)
        div_loss, div_stats = slot_diversity_loss(summary_tokens)

        sem_loss = zero
        summary_logits = None
        summary_repr = summary_tokens[:, -1].mean(dim=1)
        if self.psd_sem_head_enable:
            summary_logits = self.summary_sem_head(summary_repr)
            if target is not None:
                if self.psd_sem_loss_type == 'hard_label_ce':
                    sem_loss = self.sem_loss_fn(summary_logits, target)
                elif self.psd_sem_loss_type == 'soft_main_kl':
                    if main_logits is None:
                        raise ValueError('main_logits are required when psd_sem_loss_type=soft_main_kl')
                    sem_loss = self._soft_semantic_alignment_loss(summary_logits, main_logits)
                else:
                    raise ValueError(f'Unsupported psd_sem_loss_type: {self.psd_sem_loss_type}')

        stats = {
            'summary_abs_mean': summary_tokens.abs().mean().detach(),
            'task_target_std': pred_target_tokens.std(dim=-1, unbiased=False).mean().detach(),
            'task_target_delta_abs_mean': (
                (pred_target_tokens[:, 1:] - pred_target_tokens[:, :-1]).abs().mean().detach()
                if pred_target_tokens.size(1) > 1
                else zero.detach()
            ),
            **pred_stats,
            **reg_stats,
            **div_stats,
        }
        if summary_logits is not None:
            stats['summary_sem_conf'] = summary_logits.softmax(dim=-1).max(dim=-1)[0].mean().detach()
            if target is not None:
                stats['summary_sem_acc'] = (summary_logits.argmax(dim=-1) == target).float().mean().detach()
            if main_logits is not None:
                stats['summary_sem_agree'] = (summary_logits.argmax(dim=-1) == main_logits.argmax(dim=-1)).float().mean().detach()

        return {
            'summary_tokens': summary_tokens,
            'aux_target_tokens': pred_target_tokens,
            'aux_target_name': pred_target_name,
            'summary_repr': summary_repr,
            'summary_logits': summary_logits,
            'aux_losses': {'pred': pred_loss, 'reg': reg_loss, 'div': div_loss, 'sem': sem_loss},
            'aux_stats': stats,
        }

    def forward(
        self,
        input: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        return_sequence: bool = False,
    ):
        xyzs, features, frame_indices = self.tube_embedding(input)
        xyzs, features = self._order_points(xyzs, features)

        bsz, steps, num_tokens, _ = xyzs.shape
        denom = max(input.size(1) - 1, 1)
        time_values = frame_indices.float() / float(denom)
        time_values = time_values.view(1, steps, 1, 1).expand(bsz, steps, num_tokens, 1)
        xyzt = torch.cat([xyzs, time_values], dim=-1)
        embedding = features + self.pos_embedding(xyzt)
        embedding = self.emb_relu(embedding)

        seq = embedding.reshape(bsz, steps * num_tokens, self.dim)
        seq = self.mamba_blocks(seq)
        seq = seq.reshape(bsz, steps, num_tokens, self.dim)
        frame_states = self.frame_norm(seq.max(dim=2)[0])
        logits_seq = self.mlp_head(frame_states)
        logits = logits_seq[:, -1]

        if not return_aux and not return_sequence:
            return logits

        output = {'logits': logits}
        if return_sequence:
            output['logits_seq'] = logits_seq
            output['frame_indices'] = frame_indices.detach().cpu()
            output['frame_states'] = frame_states
        if return_aux and self.psd_enable:
            output.update(self._build_auxiliary(frame_states, seq, target=target, main_logits=logits))
        return output
