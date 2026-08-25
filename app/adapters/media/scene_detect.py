"""镜头切分适配器（PySceneDetect）。

**为什么不用 ffmpeg 自带的 scene 滤镜。** 最初为了少一个重依赖，这里用的是
`select='gt(scene,threshold)'`。实测发现它会漏掉硬切：ffmpeg 的场景检测只比较
**亮度平面**，因此两个亮度接近、只有颜色不同的镜头之间的切换，它完全看不见。
构造用例里红(#fc0000, Y≈75) → 绿(#007f00, Y≈75) 的硬切，即使把阈值降到 0.001
也检不出，而绿(Y≈75) → 蓝(Y≈29) 能检出。

这不是构造用例的特例。真实营销视频里同曝光下不同颜色的产品镜头切换、调色序列
内部的切换都属于这一类。漏掉的镜头边界会让分镜对齐和关键帧全部继承这个错误，
而且这种错在成片里表现为「画面和口播差了半拍」，几乎不可能追溯回切分阶段。

PySceneDetect 的 ContentDetector 比较 HSV 三个通道的逐像素差异，色度变化同样敏感。
代价是带进 opencv 与 numpy，对一个视频处理系统而言这个代价是合理的。
"""

from __future__ import annotations

from pathlib import Path

from scenedetect import ContentDetector, detect

from app.adapters.media.ffmpeg import build_shot_list
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.media import MediaMetadata, ShotList

# ContentDetector 的阈值量纲与 ffmpeg 的 0~1 不同，典型可用区间是 15~40。
# 27 是库的默认值，实测对短视频快剪偏稳；调低会把镜头内的大幅运动误判成切换。
_DEFAULT_CONTENT_THRESHOLD = 27.0

# 领域参数仍然用 0~1 表示「灵敏度」，与具体检测库解耦：换检测器时业务侧的
# scene_threshold 语义不变，只有这里的映射需要改。
_SENSITIVITY_TO_THRESHOLD_SPAN = (12.0, 45.0)


def sensitivity_to_threshold(sensitivity: float) -> float:
    """把 0~1 的灵敏度映射到 ContentDetector 的阈值。

    灵敏度高 → 阈值低 → 切得更碎。做成显式映射而不是让调用方直接传库的阈值，
    是为了让「换检测器」不需要改任何业务参数。
    """
    low, high = _SENSITIVITY_TO_THRESHOLD_SPAN
    return high - (high - low) * min(max(sensitivity, 0.0), 1.0)


class SceneDetectShotDetector:
    """实现 `app.domain.media.ShotDetector`。"""

    def detect_shots(self, source: Path, *, metadata: MediaMetadata, threshold: float) -> ShotList:
        if metadata.duration_ms <= 0:
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "视频时长为 0，无法切分镜头")

        try:
            scenes = detect(
                str(source),
                ContentDetector(threshold=sensitivity_to_threshold(threshold)),
            )
        except Exception as exc:
            # 解码失败、编码不支持等都归到存储/介质问题，交由上层决定是否重试。
            raise CapabilityError(
                ErrorCode.STORAGE_FAILURE, f"镜头切分失败：{source.name}（{exc}）"
            ) from exc

        # 只取切点，边界的规整（越界、过近、末尾残片）复用与 ffmpeg 实现相同的
        # 那段纯逻辑，保证两个检测器产出的镜头列表满足同样的不变量。
        cut_points_ms = [int(start.seconds * 1000) for start, _ in scenes[1:]]
        return build_shot_list(cut_points_ms, metadata.duration_ms)
