"""在冻结预测之后，用互斥对应更新两个位姿融合门。"""

from dataclasses import replace

import numpy as np


def split_tracks(frame, matches, references, recent=()):
    """继承历史特征的分组，已知分组冲突的当前轨迹从全部分组剔除。"""
    hashed = (np.arange(len(frame.features), dtype=np.uint32) + frame.index * 193)
    hashed = (hashed ^ (hashed >> 16)) * np.uint32(0x45D9F3B)
    bucket = ((hashed ^ (hashed >> 16)) % 5).astype(np.int8)
    folds = np.maximum(0, bucket - 2)
    inherited = np.full(len(folds), -1, dtype=np.int8)
    for match in matches:
        previous = references[match.reference].folds[match.reference_feature]
        known = inherited[match.query]
        conflicts = (known != -1) & (known != previous)
        inherited[match.query] = np.where(conflicts | (previous < 0), -2, previous)
    folds = np.where(inherited == -1, folds, inherited)
    groups = [[], [], []]
    for match in matches:
        for group in range(3):
            if group == 0 and match.reference in recent:
                continue
            mask = folds[match.query] == group
            if mask.any():
                groups[group].append(replace(match, query=match.query[mask],
                                             prior_points=match.prior_points[mask],
                                             reference_feature=match.reference_feature[mask]))
    return folds, groups


def angular_samples(frame, matches, scale):
    if not matches:
        return None
    indices = np.concatenate([match.query for match in matches])
    if len(np.unique(indices)) < 5 or len(np.unique(frame.cells[indices])) < 3:
        return None
    weights = np.bincount(indices)[indices]
    rays = frame.local_points[indices].astype(np.float64)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    points = np.concatenate([match.prior_points for match in matches]) / scale
    return points, rays, (1 / weights) / np.sum(1 / weights)


def angular_cost(points, rays, weights):
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        return np.inf
    angles = np.arctan2(np.linalg.norm(np.cross(points, rays), axis=1), (points * rays).sum(1))
    residual = angles / 0.012
    return float(weights @ np.where(residual <= 1, residual**2, 2 * residual - 1))


def gate_features(frame, matches, directions, rays):
    """构造点门和方向门各自的偏置与两项质量输入。"""
    if rays is None:
        features = np.array([[1., 0., 0.], [1., 0., 0.]])
    else:
        _, _, points, observed, weights, _, _ = rays
        indices = np.concatenate([match.query for match in matches])
        coverage = len(np.unique(frame.cells[indices])) / 24
        error = np.linalg.norm(points[:, :2] / points[:, 2:] - observed, axis=1)
        features = np.array([[1., coverage, min(1., float(weights @ error) / .03)],
                             [1., 0., 0.]])
    if directions:
        features[1, 1] = np.mean([min(1., match.support / 8) for match in directions])
        features[1, 2] = np.mean([min(1., match.scatter / .03) for match in directions])
    return features


def normal_equations(rays, directions, direction_terms):
    """为两类几何项分别生成加权前的法方程。"""
    normals = np.zeros((2, 6, 6))
    rhs = np.zeros((2, 6))
    if rays is not None:
        system, residual = rays[-2:]
        normals[0] = system.T @ system
        rhs[0] = system.T @ residual
    if directions:
        source, target, weight, direction_system = direction_terms(directions)
        normals[1] = direction_system.T @ direction_system
        rhs[1] = direction_system.T @ ((source - target) * weight).ravel()
    return normals, rhs


def solve_update(gains, normals, rhs, regularizer):
    """求解可微的六自由度高斯–牛顿增量，支持批量。"""
    import torch

    hessian = regularizer.T @ regularizer + (gains[..., None, None] * normals).sum(-3)
    gradient = (gains[..., None] * rhs).sum(-2)
    return torch.linalg.solve(hessian, -gradient[..., None]).squeeze(-1)


class OnlineWeights:
    def __init__(self, direction, *, online):
        initial = np.array([0.95, direction], dtype=np.float64)
        self.initial = np.column_stack((np.log(initial / (1 - initial)), np.zeros((2, 2))))
        self.parameters = self.initial.copy()
        self.online = online
        self.pending = None

    @staticmethod
    def gains(parameters, features):
        return 1 / (1 + np.exp(-np.sum(parameters * features, axis=1)))

    def observe(self, frame, prior, groups, directions, ray_terms, direction_terms, regularizer, solve):
        """确定本帧权重，并将通过核验的更新暂存到下一帧。"""
        applied_previous = self.pending is not None
        if applied_previous:
            self.parameters = self.pending
            self.pending = None
        rays = ray_terms(frame, groups[0])
        features = gate_features(frame, groups[0], directions, rays)
        coverage = features[0, 1]
        gains = self.gains(self.parameters, features)
        log = dict(applied_previous=applied_previous, point_gain=float(gains[0]),
                   direction_gain=float(gains[1]), triggered=False, accepted=False)
        if rays is None or frame.index % 8 or rays[0] < 16 or coverage < .25:
            return *gains, log
        scale = rays[1]
        training = angular_samples(frame, groups[1], scale)
        validation = angular_samples(frame, groups[2], scale)
        if training is None or validation is None:
            return *gains, log
        before = solve(prior, frame, groups[0], directions,
                       point_gain=gains[0], direction_gain=gains[1])
        if before is None:
            return *gains, log
        log['triggered'] = True
        if not self.online:
            return *gains, log

        normals, rhs = normal_equations(rays, directions, direction_terms)
        updated = self._gradient_step(features, normals, rhs, regularizer, training)
        if updated is None:
            return *gains, log
        proposed = self.gains(updated, features)
        after = solve(prior, frame, groups[0], directions,
                      point_gain=proposed[0], direction_gain=proposed[1])
        if after is None:
            return *gains, log

        def heldout(pose):
            correction = np.linalg.solve(pose, prior)
            points, rays, weights = validation
            projected = points @ correction[:3, :3].T + correction[:3, 3] / scale
            return angular_cost(projected, rays, weights)

        old_loss, new_loss = heldout(before[1]), heldout(after[1])
        if np.isfinite(old_loss) and np.isfinite(new_loss) and new_loss < old_loss - 1e-8:
            self.pending = updated
            log.update(accepted=True, validation_before=old_loss, validation_after=new_loss)
        return *gains, log

    def _gradient_step(self, features, normals, rhs, regularizer, training):
        import torch

        with torch.inference_mode(False), torch.enable_grad():
            params = torch.tensor(self.parameters, requires_grad=True)
            gains = (params * torch.tensor(features)).sum(1).sigmoid()
            increment = solve_update(gains, torch.tensor(np.asarray(normals)), torch.tensor(np.asarray(rhs)),
                                     torch.tensor(regularizer))
            x, y, z = increment[:3]
            zero = torch.zeros_like(x)
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            points, rays, weights = map(torch.tensor, training)
            projected = points @ torch.matrix_exp(skew).T + increment[3:]
            if torch.any(projected[:, 2] <= 0):
                return None
            cosine = (torch.nn.functional.normalize(projected, dim=1) * rays).sum(1)
            # 归一化射线的弦长在小角度时与弧度一致，零残差处可求导。
            square = (2 * (1 - cosine)).clamp_min(0) / .012**2
            loss = weights @ torch.where(square <= 1, square, 2 * square.clamp_min(1).sqrt() - 1)
            loss += .01 * ((params - torch.tensor(self.initial))**2).sum()
            update, = torch.autograd.grad(loss, params)
        if not torch.isfinite(update).all():
            return None
        change = np.clip(.05 * update.detach().numpy(), -.02, .02)
        return np.clip(self.parameters - change, self.initial - .5, self.initial + .5)
