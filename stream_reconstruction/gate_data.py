"""融合门训练缓存与 JSON 权重格式。"""

import json

import numpy as np


class GateEvidence:
    def __init__(self):
        self.rows = []

    def add(self, frame, prior, scale, features, normals, rhs):
        self.rows.append((frame, prior, scale, features, normals, rhs))

    def save(self, path, frame_count, direction_gain):
        if not self.rows:
            raise ValueError("没有可用于训练的点对应或方向约束")
        frame, prior, scale, features, normals, rhs = map(np.asarray, zip(*self.rows))
        with path.with_suffix(".npz.part").open("xb") as stream:
            np.savez_compressed(stream, version=1, frame_count=frame_count,
                                frame=frame, prior=prior, scale=scale, features=features,
                                normals=normals, rhs=rhs, direction_gain=direction_gain)
        path.with_suffix(".npz.part").replace(path)


def load_gate_weights(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("不支持的融合门权重版本")
    parameters = np.asarray(payload.get("parameters"), dtype=np.float64)
    if parameters.shape != (2, 3) or not np.isfinite(parameters).all() or np.max(abs(parameters)) > 20:
        raise ValueError("融合门权重需要两个有限的三参数向量")
    return parameters
