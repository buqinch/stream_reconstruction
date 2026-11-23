"""用已提交的历史共视点更新主干最后一层注意力门。"""

from dataclasses import replace

import cv2
import numpy as np
import torch

from .trajectory import Reference, pose_evidence
from .ttt import angular_cost, angular_samples
from .visual_context import RECENT


def transform_points(points, candidate, prior):
    relative = torch.linalg.solve(candidate, prior)
    return points @ relative[:3, :3].T + relative[:3, 3]


def partition(frame, match):
    """用棋盘式图像网格拆分同一锚内的梯度与核验点。"""
    row, col = np.divmod(frame.cells[match.query], 6)
    folds = (row + col) % 2
    groups = []
    for fold in (0, 1):
        use = folds == fold
        if use.sum() < 6:
            return None
        evidence = replace(match, query=match.query[use], prior_points=match.prior_points[use])
        samples = angular_samples(frame, [evidence], 1.)
        if samples is None:
            return None
        groups.append(samples)
    return groups


class VisualTTT:
    interval = 8

    def __init__(self, gate):
        self.gate = gate
        self.initial = gate.parameters_by_group.detach().clone()
        self.observations = {}
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def remember(self, frame, pose, context):
        self.observations[frame.index] = Reference(frame, pose)
        keep = {item.index for item in (*context.recent, *context.fixed, *context.anchors)}
        keep.update(context.pending)
        self.observations = {index: obs for index, obs in self.observations.items()
                             if index in keep}

    def update(self, frame, prior, pose, indices, replay):
        info = dict(accepted=False)
        if not pose.requires_grad:
            return info
        candidates = []
        for index in indices:
            if index > frame.index - RECENT or index not in self.observations:
                continue
            match = pose_evidence(frame, self.observations[index], self.matcher, prior)
            if match is not None:
                groups = partition(frame, match)
                if groups is not None:
                    candidates.append((len(match.query), index, groups))
        if not candidates:
            return info
        count, index, (training, validation) = max(candidates, key=lambda item: item[0])
        info.update(reference=index, matches=count, triggered=True)
        params = self.gate.parameters_by_group
        before = params.detach().clone()
        points, rays, weights = (torch.as_tensor(value, dtype=pose.dtype, device=pose.device)
                                 for value in training)
        initial_pose = torch.as_tensor(prior, dtype=pose.dtype, device=pose.device)
        normalized = torch.nn.functional.normalize(transform_points(points, pose, initial_pose), dim=1)
        residual = torch.linalg.vector_norm(normalized - rays, dim=1) / .012
        loss = weights @ torch.where(residual <= 1., residual.square(), 2 * residual - 1.)
        gradient, = torch.autograd.grad(loss, params)
        if not torch.isfinite(gradient).all() or gradient.abs().max() < 1e-9:
            return info
        proposal = before - .10 * gradient / gradient.abs().max()
        proposal = torch.maximum(torch.minimum(proposal, self.initial + 1.5), self.initial - 1.5)
        if torch.equal(before, proposal):
            return info

        accepted = False
        with torch.no_grad():
            try:
                params.copy_(proposal)
                trial_pose = replay()
                points, rays, weights = validation
                projected = transform_points(torch.as_tensor(points, dtype=pose.dtype, device=pose.device),
                                             trial_pose, initial_pose).cpu().numpy()
                loss_before = angular_cost(points, rays, weights)
                loss_after = angular_cost(projected, rays, weights)
                if np.isfinite([loss_before, loss_after]).all():
                    accepted = loss_after + 1e-6 < loss_before
                    info.update(loss_before=loss_before, loss_after=loss_after)
            finally:
                if not accepted:
                    params.copy_(before)
        info["accepted"] = accepted
        return info
