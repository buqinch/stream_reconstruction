"""在冻结前向之后，用已发布历史帧的稀疏点校正相机到世界位姿。"""

from collections import deque
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from .directions import BLOCK_FRAMES, DIRECTION_SIGMA, DirectionMemory
from .selection import Observation, mutual_matches
from .ttt import OnlineWeights, split_tracks


@dataclass
class Reference:
    observation: Observation
    pose: np.ndarray
    folds: np.ndarray | None = None


@dataclass
class PoseEvidence:
    reference: int
    query: np.ndarray
    prior_points: np.ndarray
    relative: np.ndarray
    reference_feature: np.ndarray | None = None


MAX_ROTATION = np.deg2rad(5)
MAX_TRANSLATION = 0.05
RAY_SIGMA = 0.012
REGULARIZER = np.diag([1 / np.deg2rad(2)] * 3 + [1 / 0.02] * 3)
SUMMARY_ROTATION_TOLERANCE = np.deg2rad(2)


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
    return PoseEvidence(reference.observation.index, q[indices], prior_points, relative, t[indices])


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


def direction_terms(directions, gain=1.):
    source = np.stack([match.prior_axis for match in directions])
    target = np.stack([match.local_axis for match in directions])
    support = np.array([match.support for match in directions])
    weight = np.sqrt(gain * np.minimum(support / 3, 1) / len(directions))[:, None]
    weight /= DIRECTION_SIGMA
    rotation_jacobian = np.cross(np.eye(3)[None], source[:, None]).transpose(0, 2, 1)
    system = np.zeros((3 * len(directions), 6))
    system[:, :3] = (rotation_jacobian * weight[:, :, None]).reshape(-1, 3)
    return source, target, weight, system


def joint_step(prior, current, matches, directions=(), *, direction_gain=1., point_gain=1.):
    """合并点重投影、旋转方向与局部先验，求一次高斯–牛顿增量。"""
    rays = ray_terms(current, matches)
    if rays is None and not directions:
        return None
    regularizer = REGULARIZER
    systems = [regularizer]
    residuals = [np.zeros(6)]
    if rays is not None:
        count, scale, points, observed, weights, system, residual = rays
        systems.insert(0, system * np.sqrt(point_gain))
        residuals.insert(0, residual * np.sqrt(point_gain))
    else:
        count, scale = 0, 1.
    if directions:
        source, target, weight, system = direction_terms(directions, direction_gain)
        systems.append(system)
        residuals.append(((source - target) * weight).ravel())

    delta = np.linalg.lstsq(np.vstack(systems), -np.concatenate(residuals), rcond=None)[0]
    if np.linalg.norm(delta[:3]) > MAX_ROTATION or np.linalg.norm(delta[3:]) > MAX_TRANSLATION:
        return None
    rotation = cv2.Rodrigues(delta[:3])[0]
    initial, trial = 0., np.linalg.norm(regularizer @ delta)**2
    if rays is not None:
        initial += point_gain * ray_cost(points, observed, weights)
        trial += point_gain * ray_cost(points @ rotation.T + delta[3:], observed, weights)
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


def summary_alignment(frame, prior, matches, method):
    """将子图写入旋转限定在通过 PnP 检查的较早子图参考上。"""
    block = frame // BLOCK_FRAMES
    rotation = prior[:3, :3]
    if block == 0:
        return rotation
    earlier = [match for match in matches if match.reference // BLOCK_FRAMES < block]
    if not earlier:
        return None
    if method == 'single':
        best = max(earlier, key=lambda match: len(match.query))
        return rotation @ best.relative[:3, :3]

    # 在共同的先验相机系中选择成对角度相容的参考。各增量已被限定在 5° 内。
    rotations = [match.relative[:3, :3] for match in earlier]
    support = np.array([len(match.query) for match in earlier])
    blocks = [match.reference // BLOCK_FRAMES for match in earlier]
    chosen, best_support = None, 0
    for left, right in combinations(range(len(earlier)), 2):
        if blocks[left] == blocks[right]:
            continue
        angle = np.linalg.norm(cv2.Rodrigues(rotations[left].T @ rotations[right])[0])
        pair_support = min(support[left], support[right])
        if angle <= SUMMARY_ROTATION_TOLERANCE and pair_support > best_support:
            chosen, best_support = [left, right], pair_support
    if chosen is None:
        return None
    increments = np.stack([cv2.Rodrigues(rotations[i])[0].ravel() for i in chosen])
    consensus = np.average(increments, axis=0, weights=support[chosen])
    return rotation @ cv2.Rodrigues(consensus)[0]


class PoseMemory:
    def __init__(self, mode, capacity, window, solver="single", summary="off", *,
                 direction_gain=1., alignment="single", ttt="off"):
        if mode not in ('anchors', 'fixed') or window < 1:
            raise ValueError('需要有效的记忆模式和正的近期窗口')
        if solver not in ('single', 'joint'):
            raise ValueError('位姿求解器须为单参考或联合求解')
        if summary != 'off' and (mode != 'fixed' or solver != 'joint'):
            raise ValueError('方向摘要需要固定帧和联合求解模式')
        if not 0 <= direction_gain <= 1:
            raise ValueError('方向能量权重须在 [0, 1] 内')
        if alignment not in ('single', 'consensus'):
            raise ValueError('摘要写入须选择单参考或跨子图共识')
        self.directions = None if summary == 'off' else DirectionMemory(summary)
        if ttt != 'off' and (ttt not in ('fixed', 'online') or summary != 'full' or not 0 < direction_gain < 1):
            raise ValueError('门控实验需要全历史摘要及 (0, 1) 内的初始方向权重')
        self.gates = None if ttt == 'off' else OnlineWeights(direction_gain, online=ttt == 'online')
        self.recent = deque(maxlen=window - 1)
        self.direction_gain = direction_gain
        self.alignment = alignment
        self.mode = mode
        self.solver = solver
        self.window = window
        self.history = deque(maxlen=capacity)
        self.fixed = []
        self.pending = {}
        self.previous_raw = self.previous_pose = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def index_payload_bytes(self):
        """仅统计已提交索引的数值数组，排除 Python 对象和暂存帧。"""
        references = {ref.observation.index: ref for ref in (*self.history, *self.fixed)}
        anchors = 0
        for ref in references.values():
            frame = ref.observation
            arrays = (frame.descriptor, frame.cells, frame.features, frame.rays,
                      frame.local_points, ref.pose, ref.folds)
            anchors += sum(array.nbytes for array in arrays if array is not None)
        summary = 0
        if self.directions is not None:
            values = self.directions
            summary = sum(array.nbytes for array in (values.moments, values.weights, values.support))
            summary += sum(array.nbytes for array in values.queries)
            summary += sum(moment.nbytes + weight.nbytes
                           for moment, weight in values.contributions.values())
        return anchors, summary

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
        directions = []
        if self.directions is not None and self.direction_gain:
            directions = self.directions.read(axes, prior[:3, :3])
        folds = None
        point_gain, direction_gain = 1., self.direction_gain
        learning = None
        if self.gates is not None:
            recent_matches = []
            for ref in list(self.recent)[-2:]:
                recent = pose_evidence(frame, ref, self.matcher, prior)
                if recent is not None:
                    recent_matches.append(recent)
            references = {ref.observation.index: ref for ref in (*self.recent, *candidates)}
            folds, groups = split_tracks(
                frame, matches + recent_matches, references,
                recent={ref.observation.index for ref in self.recent})
            matches_for_pose = groups[0]
            point_gain, direction_gain, learning = self.gates.observe(
                frame, prior, groups, directions, ray_terms, direction_terms, REGULARIZER, joint_step)
        else:
            matches_for_pose = matches
        used_directions = []
        if self.solver == 'joint':
            result = joint_step(prior, frame, matches_for_pose, directions,
                                direction_gain=direction_gain, point_gain=point_gain)
            if result is None and directions and len(matches_for_pose) >= 2:
                result = joint_step(prior, frame, matches_for_pose, point_gain=point_gain)
                directions = []
            if result is not None:
                count, pose = result
                if count:
                    used = [match.reference for match in matches_for_pose]
                used_directions = directions
        elif matches:
            best = max(matches, key=lambda match: len(match.query))
            count, pose = half_step(prior, best)
            used = [best.reference]
        # 本帧求解只读旧记忆。延迟通过的锚保留其首发时的位姿。
        entry = Reference(frame, pose.copy(), folds)
        if self.gates is not None:
            self.recent.append(entry)
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
        if learning is not None:
            log['ttt'] = learning
        if self.directions is not None:
            alignment = summary_alignment(frame.index, prior, matches, self.alignment)
            log['directions'] = [dict(slot=match.slot, support=match.support, scatter=match.scatter)
                                 for match in used_directions]
            self.directions.after_publish(frame.index, axes, alignment)
            log['summary_commit'] = self.directions.committed
            log['summary_submaps'] = len(self.directions.contributions)
        return pose, log
