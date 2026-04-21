from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_summary(summary_tokens: torch.Tensor) -> torch.Tensor:
    return summary_tokens.reshape(-1, summary_tokens.size(-1))


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class AntiCollapseRegularizer(nn.Module):
    def __init__(
        self,
        reg_type: str = 'var_cov',
        var_target: float = 1.0,
        cov_weight: float = 1.0,
        sig_floor: float = 0.25,
        sig_uniform_weight: float = 0.1,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.reg_type = reg_type
        self.var_target = var_target
        self.cov_weight = cov_weight
        self.sig_floor = sig_floor
        self.sig_uniform_weight = sig_uniform_weight
        self.eps = eps

    def forward(self, summary_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = _flatten_summary(summary_tokens)
        if z.size(0) <= 1:
            zero = summary_tokens.new_zeros(())
            return zero, {
                'summary_std': zero,
                'summary_std_min': zero,
                'summary_cov_offdiag': zero,
                'summary_sig_min': zero,
            }

        if self.reg_type == 'var_cov':
            return self._var_cov(z)
        if self.reg_type == 'sigreg':
            return self._sigreg(z)
        raise ValueError(f'Unsupported reg_type: {self.reg_type}')

    def _var_cov(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        var_loss = F.relu(self.var_target - std).pow(2).mean()

        cov = (z.T @ z) / max(z.size(0) - 1, 1)
        offdiag = _off_diagonal(cov)
        cov_loss = offdiag.pow(2).mean() if offdiag.numel() > 0 else cov.new_zeros(())
        loss = var_loss + self.cov_weight * cov_loss
        stats = {
            'summary_std': std.mean().detach(),
            'summary_std_min': std.min().detach(),
            'summary_cov_offdiag': offdiag.abs().mean().detach() if offdiag.numel() > 0 else cov.new_zeros(()),
            'summary_sig_min': cov.diag().min().sqrt().detach(),
        }
        return loss, stats

    def _sigreg(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / max(z.size(0) - 1, 1)
        eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        eigvals = torch.linalg.eigvalsh(cov + self.eps * eye)
        sing_vals = torch.sqrt(torch.clamp(eigvals, min=self.eps))
        floor_loss = F.relu(self.sig_floor - sing_vals).pow(2).mean()
        uniform_loss = torch.var(torch.log(sing_vals + self.eps), unbiased=False)
        loss = floor_loss + self.sig_uniform_weight * uniform_loss
        stats = {
            'summary_std': sing_vals.mean().detach(),
            'summary_std_min': sing_vals.min().detach(),
            'summary_cov_offdiag': _off_diagonal(cov).abs().mean().detach() if cov.size(0) > 1 else cov.new_zeros(()),
            'summary_sig_min': sing_vals.min().detach(),
        }
        return loss, stats


def slot_diversity_loss(summary_tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    num_slots = summary_tokens.size(-2)
    if num_slots <= 1:
        zero = summary_tokens.new_zeros(())
        return zero, {'slot_corr': zero, 'slot_max_corr': zero}
    slots = F.normalize(summary_tokens, dim=-1)
    gram = torch.matmul(slots, slots.transpose(-1, -2))
    eye = torch.eye(num_slots, device=gram.device, dtype=gram.dtype).view(1, 1, num_slots, num_slots)
    offdiag = gram - eye
    loss = offdiag.pow(2).mean()
    stats = {
        'slot_corr': offdiag.abs().mean().detach(),
        'slot_max_corr': offdiag.abs().max().detach(),
    }
    return loss, stats
