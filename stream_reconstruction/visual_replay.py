"""从四锚运行的已完成轨迹读取因果回放记录。"""

from dataclasses import dataclass
import json

from .selection import Decision


@dataclass(frozen=True)
class VisualReplay:
    anchors: tuple
    decisions: dict

    @classmethod
    def load(cls, directory, paths, lookahead):
        with (directory / "trajectory.jsonl").open(encoding="utf-8") as stream:
            rows = tuple(map(json.loads, stream))
        with (directory / "decisions.jsonl").open(encoding="utf-8") as stream:
            decisions = tuple(map(json.loads, stream))
        if len(rows) != len(paths) or len(decisions) != len(paths):
            raise ValueError("锚回放须覆盖全部输入帧")

        anchors = []
        confirmed = {}
        for index, (path, row, decision) in enumerate(zip(paths, rows, decisions)):
            if (row["index"] != index or row["image"] != path.name
                    or row["visual_mode"] != "anchors4"
                    or decision["candidate"] != index or decision["image"] != path.name):
                raise ValueError("锚回放与输入帧或四锚模式不一致")
            selected = tuple(row["visual_context"]["anchors"])
            if (len(selected) > 4 or len(set(selected)) != len(selected)
                    or any(type(anchor) is not int or not 0 <= anchor <= index - 12
                           for anchor in selected)):
                raise ValueError(f"第 {index} 帧的回放锚不满足因果窗口")
            anchors.append(selected)
            when = decision["decided_at"]
            expected = min(index + lookahead, len(paths) - 1)
            if when != expected or (index + lookahead >= len(paths) and decision["kept"]):
                raise ValueError("锚回放与观察步数不一致")
            if index + lookahead < len(paths):
                info = {key: value for key, value in decision.items() if key != "image"}
                confirmed[when] = Decision(**info)
        return cls(tuple(anchors), confirmed)
