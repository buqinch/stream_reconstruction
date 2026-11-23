"""最后一层跨帧注意力的可训练历史融合门。"""

import json

import torch


class VisualHistoryGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_by_group = torch.nn.Parameter(torch.tensor([[-4., 0.]] * 3))

    def forward(self, similarity):
        scores = self.parameters_by_group[:, 0]
        scores = scores + self.parameters_by_group[:, 1] * (similarity - .75)
        return .25 * scores.sigmoid()

    @classmethod
    def load(cls, path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("不支持的视觉门版本")
        try:
            params = torch.as_tensor(payload["parameters"], dtype=torch.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("缺少有效的视觉门参数") from error
        if params.shape != (3, 2) or not torch.isfinite(params).all() or params.abs().max() > 20:
            raise ValueError("视觉门需要三个有限的双参数向量")
        gate = cls().requires_grad_(False)
        gate.parameters_by_group.copy_(params)
        return gate
