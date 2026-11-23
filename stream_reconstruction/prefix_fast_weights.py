"""以开头十二帧的视觉键值对拟合会话级快权重。"""

import math

import torch
from torch.nn.functional import normalize

from abot_recon.modeling.streaming.rope3d import apply_rotary_emb


class PrefixFastWeights:
    frames = 12
    holdout = 2
    gain = .05
    ridge = .002

    def __init__(self, rope):
        self.rope = rope
        self.samples = []
        self.weights = self.mean = None
        self.calibration = None

    def features(self, tokens, index):
        # 去除绝对时间相位，让后续时刻仍可查询前缀，保留空间旋转。
        time_dim = self.rope.fhw_dim[0]
        phase = self.rope.freqs[index, :time_dim // 2].conj()
        temporal = apply_rotary_emb(tokens[..., :time_dim].float(), phase)
        neutral = torch.cat((temporal, tokens[..., time_dim:].float()), dim=-1)
        return normalize(neutral.transpose(0, 1), dim=-1)

    def fit(self, keys, values):
        mean = values.mean(dim=1, keepdim=True)
        transpose = keys.transpose(-1, -2)
        identity = torch.eye(keys.shape[-1], device=keys.device)
        covariance = transpose @ keys + self.ridge * keys.shape[1] * identity
        weights = torch.linalg.solve(covariance, transpose @ (values - mean))
        return weights, mean

    def collect(self, cache, index):
        keys, values = cache.unbind(0)
        stride = max(1, (len(keys) + 95) // 96)
        self.samples.append((self.features(keys[::stride], index),
                             values[::stride].float().transpose(0, 1).contiguous()))
        if len(self.samples) == self.frames:
            self.calibrate()

    def calibrate(self):
        keys, values = (torch.cat(group, dim=1) for group in zip(*self.samples))
        count = sum(pair[0].shape[1] for pair in self.samples[:-self.holdout])
        weights, mean = self.fit(keys[:, :count], values[:, :count])
        held_keys, held_values = keys[:, count:], values[:, count:]
        baseline = float((held_values - mean).square().mean())
        fitted = float((held_values - mean - held_keys @ weights).square().mean())
        accepted = math.isfinite(baseline) and math.isfinite(fitted) and fitted < baseline
        if accepted:
            candidate, center = self.fit(keys, values)
            accepted = bool(torch.isfinite(candidate).all() and torch.isfinite(center).all())
            if accepted:
                self.weights, self.mean = candidate, center
        self.calibration = dict(accepted=accepted, fit_tokens=count,
                                holdout_tokens=keys.shape[1] - count,
                                baseline_mse=baseline if math.isfinite(baseline) else None,
                                fitted_mse=fitted if math.isfinite(fitted) else None)
        self.samples.clear()

    def read(self, query, attended, index):
        recovered = self.features(query, index) @ self.weights + self.mean
        return torch.lerp(attended, recovered.transpose(0, 1).to(attended.dtype), self.gain)
