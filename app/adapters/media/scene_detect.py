"""镜头切分适配器（PySceneDetect）。

**为什么不用 ffmpeg 自带的 scene 滤镜。** 最初为了少一个重依赖，这里用的是
`select='gt(scene,threshold)'`。实测发现它会漏掉硬切：ffmpeg 的场景检测只比较
**亮度平面**，因此两个亮度接近、只有颜色不同的镜头之间的切换，它完全看不见。
构造用例里红(#fc0000, Y≈75) → 绿(#007f00, Y≈75) 的硬切，即使把阈值降到 0.001
也检不出，而绿(Y≈75) → 蓝(Y≈29) 能检出。

这不是构造用例的特例。真实营销视频里同曝光下不同颜色的产品镜头切换、调色序列
内部的切换都属于这一类。漏掉的镜头边界会让分镜对齐和关键帧全部继承这个错误，
而且这种错在成片里表现为「画面和口播差了半拍」，几乎不可能追溯回切分阶段。

**为什么用 AdaptiveDetector 而不是 ContentDetector。** ContentDetector 用固定阈值
比较 HSV 三通道差异，并把 hue/sat/lum 等权平均。于是纯色相切换（绿 #007f00 →
深蓝 #000080：色相差 60，饱和度与明度均无变化）得分只有 60/3 = 20，低于库默认
阈值 27，检不出。把阈值调到能检出它（15）之后，一段持续旋转且色相漂移的**单一
连续镜头**会被误切成 4 段。也就是说固定阈值只能在「漏切」和「误切」之间取舍。

AdaptiveDetector 比较的是当前帧与邻域滚动均值的偏离，因此渐变的运动不会累积成
切点，而突变仍然显著。实测在同一组素材上两者兼顾：纯色相硬切全部检出，运动素材
零误切。

代价是带进 opencv 与 numpy，对一个视频处理系统而言这个代价是合理的。
"""

from __future__ import annotations

from pathlib import Path

from scenedetect import AdaptiveDetector, detect

from app.adapters.media.ffmpeg import build_shot_list
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.media import MediaMetadata, ShotList

# 领域参数用 0~1 的「灵敏度」表达，与具体检测库解耦：换检测器时业务侧参数语义
# 不变，只有这里的映射需要改。
#
# 区间端点的选取：AdaptiveDetector 的 adaptive_threshold 越低越敏感，默认 3.0。
# 区间被刻意设计成**中点 0.5 恰好落在库默认值**——默认值必须是一个已实测可用的
# 点，而不是区间上的任意位置。第一版没做到这一点：区间定成 12~45、领域默认取
# 0.3，映射结果比库自身默认值还不敏感，于是漏切。
_SENSITIVITY_SPAN = (1.0, 5.0)


def sensitivity_to_threshold(sensitivity: float) -> float:
    """把 0~1 的灵敏度映射到 ContentDetector 的阈值。

    灵敏度高 → 阈值低 → 切得更碎。0.5 对应库默认值 27。
    """
    low, high = _SENSITIVITY_SPAN
    return high - (high - low) * min(max(sensitivity, 0.0), 1.0)


class SceneDetectShotDetector:
    """实现 `app.domain.media.ShotDetector`。"""

    def detect_shots(
        self, source: Path, *, metadata: MediaMetadata, sensitivity: float
    ) -> ShotList:
        if metadata.duration_ms <= 0:
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "视频时长为 0，无法切分镜头")

        try:
            scenes = detect(
                str(source),
                AdaptiveDetector(adaptive_threshold=sensitivity_to_threshold(sensitivity)),
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
