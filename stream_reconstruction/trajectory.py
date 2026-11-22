"""在冻结前向之后，用已发布历史帧的稀疏点校正相机到世界位姿。"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .directions import BLOCK_FRAMES, DIRECTION_SIGMA, DirectionMemory
from .selection import Observation, mutual_matches


@dataclass
class Reference:
    observation: Observation
    pose: np.ndarray


@dataclass
class PoseEvidence:
    reference: int
    query: np.ndarray
    prior_points: np.ndarray
    relative: np.ndarray


MAX_ROTATION = np.deg2rad(5)
MAX_TRANSLATION = 0.05
RAY_SIGMA = 0.012


def pose_evidence(current, reference, matcher, prior):
    """用单参考 PnP 核验点对应，同时保留联合求解所需的内点。"""
    pairs = mutual_matches(current, reference.observation, matcher)
    if len(pairs) < 12:
        return None
    q = np.array([m.queryIdx for m in pairs])
    t = np.array([m.trainIdx for m in pairs])
    points = reference.observation.local_points[t].astype(np.float64)
    local = current.local_points[q].astype(np.float64)
    image = np.ascontiguousarray(local[:, :2] / local[:, 2:])
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        points, image, np.eye(3), None, flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=RAY_SIGMA, iterationsCount=100)
    if not ok or inliers is None or len(inliers) < max(10, len(q) // 2):
        return None
    indices = inliers.ravel()
    if min(len(np.unique(current.cells[q[indices]])),
           len(np.unique(reference.observation.cells[t[indices]]))) < 4:
        return None
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.ravel()
    projected = points[indices] @ transform[:3, :3].T + translation.ravel()
    if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
        return None
    ratio = np.median(projected[:, 2] / local[indices, 2])
    if not 0.8 < ratio < 1.25:
        return None
    target = reference.pose @ np.linalg.inv(transform)
    relative = np.linalg.solve(prior, target)
    delta = cv2.Rodrigues(relative[:3, :3])[0]
    if np.linalg.norm(delta) > MAX_ROTATION:
        return None
    if np.linalg.norm(relative[:3, 3]) > MAX_TRANSLATION * np.median(local[indices, 2]):
        return None
    to_prior = np.linalg.solve(prior, reference.pose)
    prior_points = points[indices] @ to_prior[:3, :3].T + to_prior[:3, 3]
    return PoseEvidence(reference.observation.index, q[indices], prior_points, relative)


def half_step(prior, evidence):
    relative = evidence.relative.copy()
    # 固定半步融合，用于与冻结轨迹对照。
    delta = cv2.Rodrigues(relative[:3, :3])[0]
    relative[:3, :3] = cv2.Rodrigues(0.5 * delta)[0]
    relative[:3, 3] *= 0.5
    return len(evidence.query), prior @ relative


def estimate_pose(current, reference, matcher, prior):
    evidence = pose_evidence(current, reference, matcher, prior)
    return None if evidence is None else half_step(prior, evidence)


def ray_terms(current, matches):
    """构造按不同当前特征归一化的重投影线性系统。"""
    if len(matches) < 2:
        return None
    queries = np.concatenate([match.query for match in matches])
    unique, inverse, counts = np.unique(queries, return_inverse=True, return_counts=True)
    local = current.local_points[queries].astype(np.float64)
    observed = local[:, :2] / local[:, 2:]
    scale = np.median(current.local_points[unique, 2])
    points = np.concatenate([match.prior_points for match in matches]) / scale
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        return None

    origin = np.zeros(3)
    projected, jacobian = cv2.projectPoints(points, origin, origin, np.eye(3), None)
    residual = (projected[:, 0] - observed) / RAY_SIGMA
    error = np.linalg.norm(residual, axis=1)
    # 同一当前特征跨参考重复出现时，共享该特征的观测权重。
    weights = 1.0 / (len(unique) * counts[inverse])
    robust = np.sqrt(weights / np.maximum(1.0, error)).repeat(2)
    system = (jacobian[:, :6] / RAY_SIGMA) * robust[:, None]
    singular = np.linalg.svd(system, compute_uv=False)
    if singular[-1] <= 1e-3 * singular[0]:
        return None
    return len(unique), scale, points, observed, weights, system, residual.ravel() * robust


def ray_cost(points, observed, weights):
    if np.any(points[:, 2] <= 0):
        return np.inf
    error = np.linalg.norm((points[:, :2] / points[:, 2:] - observed) / RAY_SIGMA, axis=1)
    return weights @ np.where(error <= 1, error**2, 2 * error - 1)


def joint_step(prior, current, matches, directions=()):
    """合并点重投影、旋转方向与局部先验，求一次高斯–牛顿增量。"""
    rays = ray_terms(current, matches)
    if rays is None and not directions:
        return None
    regularizer = np.diag([1 / np.deg2rad(2)] * 3 + [1 / 0.02] * 3)
    systems = [regularizer]
    residuals = [np.zeros(6)]
    if rays is not None:
        count, scale, points, observed, weights, system, residual = rays
        systems.insert(0, system)
        residuals.insert(0, residual)
    else:
        count, scale = 0, 1.
    if directions:
        source = np.stack([match.prior_axis for match in directions])
        target = np.stack([match.local_axis for match in directions])
        support = np.array([match.support for match in directions])
        weight = np.sqrt(np.minimum(support / 3, 1) / len(directions))[:, None]
        weight /= DIRECTION_SIGMA
        rotation_jacobian = np.cross(np.eye(3)[None], source[:, None]).transpose(0, 2, 1)
        system = np.zeros((3 * len(directions), 6))
        system[:, :3] = (rotation_jacobian * weight[:, :, None]).reshape(-1, 3)
        systems.append(system)
        residuals.append(((source - target) * weight).ravel())

    delta = np.linalg.lstsq(np.vstack(systems), -np.concatenate(residuals), rcond=None)[0]
    if np.linalg.norm(delta[:3]) > MAX_ROTATION or np.linalg.norm(delta[3:]) > MAX_TRANSLATION:
        return None
    rotation = cv2.Rodrigues(delta[:3])[0]
    initial, trial = 0., np.linalg.norm(regularizer @ delta)**2
    if rays is not None:
        initial += ray_cost(points, observed, weights)
        trial += ray_cost(points @ rotation.T + delta[3:], observed, weights)
    if directions:
        initial += np.linalg.norm((source - target) * weight)**2
        trial += np.linalg.norm((source @ rotation.T - target) * weight)**2
    if trial >= initial:
        return None
    # 优化增量从先验相机系映射到当前相机系，发布位姿需要其逆。
    correction = np.eye(4)
    correction[:3, :3] = rotation.T
    correction[:3, 3] = -scale * (rotation.T @ delta[3:])
    return count, prior @ correction


class PoseMemory:
    def __init__(self, mode, capacity, window, solver="single", summary="off"):
        if mode not in ('anchors', 'fixed') or window < 1:
            raise ValueError('需要有效的记忆模式和正的近期窗口')
        if solver not in ('single', 'joint'):
            raise ValueError('位姿求解器须为单参考或联合求解')
        if summary != 'off' and (mode != 'fixed' or solver != 'joint'):
            raise ValueError('方向摘要需要固定帧和联合求解模式')
        self.directions = None if summary == 'off' else DirectionMemory(summary)
        self.mode = mode
        self.solver = solver
        self.window = window
        self.history = deque(maxlen=capacity)
        self.fixed = []
        self.pending = {}
        self.previous_raw = self.previous_pose = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def advance(self, frame, raw, decision, axes=()):
        prior = raw.astype(np.float64)
        if self.previous_pose is not None:
            prior = self.previous_pose @ np.linalg.solve(self.previous_raw, prior)
        cutoff = frame.index - self.window
        fixed = [r for r in self.fixed if r.observation.index <= cutoff]
        fixed_ids = {r.observation.index for r in fixed}
        eligible = [r for r in self.history
                    if r.observation.index <= cutoff and r.observation.index not in fixed_ids]
        eligible.sort(key=lambda r: float(r.observation.descriptor @ frame.descriptor),
                      reverse=True)
        candidates = eligible[:6] + fixed
        matches = []
        for ref in candidates:
            evidence = pose_evidence(frame, ref, self.matcher, prior)
            if evidence is not None:
                matches.append(evidence)
        pose, count, used = prior, 0, []
        directions = [] if self.directions is None else self.directions.read(axes, prior[:3, :3])
        used_directions = []
        if self.solver == 'joint':
            result = joint_step(prior, frame, matches, directions)
            if result is None and directions and len(matches) >= 2:
                result = joint_step(prior, frame, matches)
                directions = []
            if result is not None:
                count, pose = result
                if count:
                    used = [match.reference for match in matches]
                used_directions = directions
        elif matches:
            best = max(matches, key=lambda match: len(match.query))
            count, pose = half_step(prior, best)
            used = [best.reference]
        # 本帧求解只读旧记忆。延迟通过的锚保留其首发时的位姿。
        entry = Reference(frame, pose.copy())
        self.pending[frame.index] = entry
        if self.mode == 'fixed' and frame.index < 3:
            self.fixed.append(entry)
        if decision is not None:
            resolved = self.pending.pop(decision.candidate)
            if decision.kept:
                self.history.append(resolved)
        self.previous_raw = raw.astype(np.float64)
        self.previous_pose = pose.copy()
        log = dict(candidates=[r.observation.index for r in candidates],
                   reference=used[0] if len(used) == 1 else None,
                   references=used, verified=[match.reference for match in matches],
                   inliers=count)
        if self.directions is not None:
            block = frame.index // BLOCK_FRAMES
            earlier = [match for match in matches if match.reference // BLOCK_FRAMES < block]
            alignment = prior[:3, :3] if block == 0 else None
            if earlier:
                best = max(earlier, key=lambda match: len(match.query))
                alignment = prior[:3, :3] @ best.relative[:3, :3]
            log['directions'] = [dict(slot=match.slot, support=match.support, scatter=match.scatter)
                                 for match in used_directions]
            self.directions.after_publish(frame.index, axes, alignment)
            log['summary_commit'] = self.directions.committed
            log['summary_submaps'] = len(self.directions.contributions)
        return pose, log
