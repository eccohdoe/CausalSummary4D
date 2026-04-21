import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils


class CausalP4DConv(nn.Module):
    """Strict-causal temporal point convolution for MSR recognition.

    Compared with the original symmetric temporal window, each anchor timestep t
    only aggregates frames from [t-k+1, ..., t]. Missing history is handled by
    left-replicating the first frame while keeping negative temporal offsets.

    Output feature layout is unified to (B, T_out, N_anchor, C) to simplify the
    causal backbone implementation.
    """

    def __init__(
        self,
        in_planes: int,
        mlp_planes: List[int],
        mlp_batch_norm: List[bool],
        mlp_activation: List[bool],
        spatial_kernel_size: List[float],
        spatial_stride: int,
        temporal_kernel_size: int,
        temporal_stride: int = 1,
        operator: str = '+',
        spatial_pooling: str = 'max',
        temporal_pooling: str = 'mean',
        bias: bool = False,
    ) -> None:
        super().__init__()
        assert temporal_kernel_size >= 1 and temporal_kernel_size % 2 == 1
        self.in_planes = in_planes
        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling
        self.out_planes = mlp_planes[-1]

        conv_d = [nn.Conv2d(4, mlp_planes[0], kernel_size=1, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_planes, mlp_planes[0], kernel_size=1, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)
        else:
            self.conv_f = None

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(mlp_planes[i - 1], mlp_planes[i], kernel_size=1, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)

    def _prepare_feature_list(self, features: Optional[torch.Tensor]) -> Optional[list]:
        if features is None:
            return None
        if features.dim() != 4:
            raise ValueError(f'Expected features with 4 dims, got {features.shape}')
        # Accept either (B, T, C, N) or (B, T, N, C).
        if features.shape[2] == self.in_planes:
            feat = features
        elif features.shape[3] == self.in_planes:
            feat = features.permute(0, 1, 3, 2).contiguous()
        else:
            raise ValueError(f'Cannot infer feature layout for shape {features.shape}')
        feat = torch.split(feat, split_size_or_sections=1, dim=1)
        return [torch.squeeze(x, dim=1).contiguous() for x in feat]

    def _spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_pooling == 'max':
            return torch.max(x, dim=-1, keepdim=False)[0]
        if self.spatial_pooling == 'sum':
            return torch.sum(x, dim=-1, keepdim=False)
        return torch.mean(x, dim=-1, keepdim=False)

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_pooling == 'max':
            return torch.max(x, dim=1, keepdim=False)[0]
        if self.temporal_pooling == 'sum':
            return torch.sum(x, dim=1, keepdim=False)
        return torch.mean(x, dim=1, keepdim=False)

    def forward(
        self,
        xyzs: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            xyzs: (B, T, N, 3)
            features: optional (B, T, C, N) or (B, T, N, C)
        Returns:
            new_xyzs: (B, T_out, N_anchor, 3)
            new_features: (B, T_out, N_anchor, C)
            frame_indices: (T_out,) raw input frame indices used as anchors
        """
        device = xyzs.device
        nframes = xyzs.size(1)
        npoints = xyzs.size(2)
        xyz_list = torch.split(xyzs, split_size_or_sections=1, dim=1)
        xyz_list = [torch.squeeze(x, dim=1).contiguous() for x in xyz_list]
        feat_list = self._prepare_feature_list(features)

        new_xyzs = []
        new_features = []
        frame_indices = []
        history = self.temporal_kernel_size - 1

        for anchor_t in range(0, nframes, self.temporal_stride):
            anchor_idx = pointnet2_utils.furthest_point_sample(xyz_list[anchor_t], npoints // self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(
                xyz_list[anchor_t].transpose(1, 2).contiguous(), anchor_idx
            )
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()
            anchor_xyz_expanded = anchor_xyz_flipped.unsqueeze(3)

            temporal_features = []
            for offset in range(-history, 1):
                src_t = max(anchor_t + offset, 0)
                neighbor_xyz = xyz_list[src_t]
                idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(
                    neighbor_xyz.transpose(1, 2).contiguous(), idx
                )

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded
                t_displacement = torch.full(
                    (xyz_displacement.size(0), 1, xyz_displacement.size(2), xyz_displacement.size(3)),
                    float(offset),
                    device=device,
                    dtype=xyz_displacement.dtype,
                )
                displacement = self.conv_d(torch.cat([xyz_displacement, t_displacement], dim=1))

                if feat_list is not None:
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(feat_list[src_t], idx)
                    feature = self.conv_f(neighbor_feature_grouped)
                    feature = feature + displacement if self.operator == '+' else feature * displacement
                else:
                    feature = displacement

                feature = self.mlp(feature)
                feature = self._spatial_pool(feature)  # (B, C, N_anchor)
                temporal_features.append(feature)

            temporal_features = torch.stack(temporal_features, dim=1)
            temporal_features = self._temporal_pool(temporal_features).transpose(1, 2).contiguous()
            new_xyzs.append(anchor_xyz)
            new_features.append(temporal_features)
            frame_indices.append(anchor_t)

        return (
            torch.stack(new_xyzs, dim=1),
            torch.stack(new_features, dim=1),
            torch.tensor(frame_indices, device=device, dtype=torch.long),
        )
