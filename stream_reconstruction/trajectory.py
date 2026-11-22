"""在冻结前向之后，用已发布历史帧的稀疏点校正相机到世界位姿。"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .selection import Observation, mutual_matches


@dataclass
class Reference:
    observation: Observation
    pose: np.ndarray


def estimate_pose(current, reference, matcher, prior):
    """归一化像面 PnP，并检查内点覆盖、正深度和相对于先验的更新幅度。"""
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
        reprojectionError=0.012, iterationsCount=100)
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
    if np.linalg.norm(delta) > np.deg2rad(5):
        return None
    if np.linalg.norm(relative[:3, 3]) > 0.05 * np.median(local[indices, 2]):
        return None
    # 固定半步融合，用于与冻结轨迹对照。
    relative[:3, :3] = cv2.Rodrigues(0.5 * delta)[0]
    relative[:3, 3] *= 0.5
    return len(indices), prior @ relative


class PoseMemory:
    def __init__(self, mode, capacity, window):
        if mode not in ('anchors', 'fixed') or window < 1:
            raise ValueError('需要有效的记忆模式和正的近期窗口')
        self.mode = mode
        self.window = window
        self.history = deque(maxlen=capacity)
        self.fixed = []
        self.pending = {}
        self.previous_raw = self.previous_pose = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def advance(self, frame, raw, decision):
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
        pose = prior
        best_count, best_ref = 0, None
        for ref in candidates:
            result = estimate_pose(frame, ref, self.matcher, prior)
            if result is not None and result[0] > best_count:
                best_count, pose = result
                best_ref = ref.observation.index
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
        return pose, dict(candidates=[r.observation.index for r in candidates],
                          reference=best_ref, inliers=best_count)
