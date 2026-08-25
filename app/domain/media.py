"""媒体处理的领域类型与协议（设计文档 6.2）。

视频预处理的产物是后面所有环节的地基：镜头边界决定分镜怎么对齐，关键帧决定
视频理解看到什么，音轨决定转写从哪来。所以这些类型定义在 domain，而具体用
ffmpeg 还是别的工具是适配层的事。

一个刻意的取舍：**镜头切分只输出边界，不做任何语义判断。** 「这一段是钩子还是
证明」属于视频理解（6.3）的职责。混在一起会让「切分错了」和「理解错了」两类
问题无法分辨。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MediaMetadata(BaseModel):
    """媒体基础信息（设计文档 6.2「计算视频基础信息」）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    duration_ms: Annotated[int, Field(ge=0)]
    width: Annotated[int, Field(ge=0)]
    height: Annotated[int, Field(ge=0)]
    fps: Annotated[float, Field(ge=0)]
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool

    @property
    def aspect_ratio(self) -> str:
        """归一化的比例字符串，用于平台规格校验（T6-2）。"""
        if self.width == 0 or self.height == 0:
            return "unknown"
        from math import gcd

        divisor = gcd(self.width, self.height)
        return f"{self.width // divisor}:{self.height // divisor}"

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


class Shot(BaseModel):
    """一个镜头的时间边界。

    只有边界，没有语义标注——语义是视频理解的职责。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: Annotated[int, Field(ge=0)]
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"镜头 {self.index} 的结束时间必须晚于开始时间")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def midpoint_ms(self) -> int:
        """取中点抽帧，而不是取首帧。

        首帧常常正好落在转场上，抽出来是黑帧或叠化的糊图，对视频理解毫无价值。
        """
        return self.start_ms + self.duration_ms // 2


class ShotList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shots: tuple[Shot, ...]

    @model_validator(mode="after")
    def _shots_are_contiguous_and_ordered(self) -> Self:
        """镜头必须首尾相接且有序。

        允许空隙或重叠会让下游分镜对齐出现无法定位的偏移，而那种偏移在成片里
        表现为「口播和画面差半秒」，排查时几乎不可能追回是切分阶段的问题。
        """
        for position, shot in enumerate(self.shots):
            if shot.index != position:
                raise ValueError(f"镜头序号不连续：位置 {position} 的 index 为 {shot.index}")
            if position > 0 and shot.start_ms != self.shots[position - 1].end_ms:
                raise ValueError(f"镜头 {shot.index} 与上一镜头之间存在空隙或重叠")
        return self

    @property
    def count(self) -> int:
        return len(self.shots)

    @property
    def total_duration_ms(self) -> int:
        return self.shots[-1].end_ms - self.shots[0].start_ms if self.shots else 0


class MediaProbe(Protocol):
    """读取媒体信息。"""

    def probe(self, path: Path) -> MediaMetadata: ...


class AudioExtractor(Protocol):
    def extract_audio(self, source: Path, target: Path) -> Path:
        """抽取音轨。返回实际写出的文件路径。"""
        ...


class ShotDetector(Protocol):
    def detect_shots(
        self, source: Path, *, metadata: MediaMetadata, sensitivity: float
    ) -> ShotList:
        """切分镜头。

        `sensitivity` 取 0~1，越大切得越碎。用「灵敏度」而不是「阈值」命名是因为
        不同检测库的阈值方向相反（有的越大越敏感，有的越小越敏感），暴露阈值必然
        被传错方向，而传错方向的症状是「镜头数不对」——很容易被当成视频本身的特性
        而不是参数问题。

        实现必须返回首尾相接、覆盖全片的镜头列表；检测不到任何切换时返回
        单个覆盖全片的镜头，而不是空列表——空列表会让下游需要到处判空。
        """
        ...


class FrameExtractor(Protocol):
    def extract_frame(self, source: Path, target: Path, *, at_ms: int) -> Path: ...
