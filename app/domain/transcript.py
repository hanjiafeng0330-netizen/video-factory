"""语音转写（设计文档 6.2）。

转写不是「一段文本」，而是**带时间戳的句子序列**。时间戳是后面一切的前提：
没有它，转写就无法和镜头对齐，而不对齐的转写对视频理解几乎没有价值——模型看到
一堆句子和一堆画面，但不知道哪句配哪个画面，也就无法判断「这个钩子是靠画面还是
靠话术立起来的」。

刻意不做的事：

- **不做说话人分离。** 一期素材以单人口播为主，引入 diarization 的收益远小于
  它带来的失败模式。
- **不做标点以外的后处理。** 转写应当是「模型说了什么」的忠实记录；纠错、改写
  属于人工修订，且必须产生新版本而不是就地覆盖原始转写。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranscriptLine(BaseModel):
    """一句转写。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: Annotated[int, Field(ge=0)]
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=1)]
    text: str = Field(min_length=1)
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None
    """模型自报的置信度。用于在审核页标出可能听错的句子，不用于自动丢弃——
    自动丢弃低置信句会让转写出现无声的空洞，比留着一句可疑的话更难排查。"""

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"转写第 {self.index} 句的结束时间必须晚于开始时间")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def overlap_ms(self, start_ms: int, end_ms: int) -> int:
        """与给定时间段的重叠时长。镜头归属就靠它计算。"""
        return max(0, min(self.end_ms, end_ms) - max(self.start_ms, start_ms))


class Transcript(BaseModel):
    """一条音轨的完整转写。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str = Field(min_length=2, max_length=16)
    lines: tuple[TranscriptLine, ...]
    model_name: str = Field(min_length=1, max_length=64)
    """用了哪个模型。换模型会改变转写结果，因此它必须进产物，否则无法回答
    「这条脚本当初依据的转写是哪个模型出的」。"""

    audio_asset_id: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _lines_are_ordered(self) -> Self:
        """按开始时间升序，序号连续。

        允许相邻句在时间上轻微重叠（ASR 断句本身有抖动），但不允许乱序——
        乱序会让后续按时间归属镜头的逻辑静默出错。
        """
        for position, line in enumerate(self.lines):
            if line.index != position:
                raise ValueError(f"转写句序号不连续：位置 {position} 的 index 为 {line.index}")
            if position > 0 and line.start_ms < self.lines[position - 1].start_ms:
                raise ValueError(f"转写句未按时间排序：第 {position} 句开始时间早于上一句")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.lines

    @property
    def full_text(self) -> str:
        return "".join(line.text for line in self.lines)

    @property
    def speech_ms(self) -> int:
        """有语音的总时长。与视频时长之比可用于区分口播型与画面型素材。"""
        return sum(line.duration_ms for line in self.lines)


class SpeechRecognizer(Protocol):
    """语音转写适配器协议。

    实现放在 `app.adapters.asr`。本地模型与云服务由同一套协议接入，业务侧不感知
    差异——这一点对 ASR 尤其重要：本地模型零调用费但慢，云服务快但按时长计费且
    素材要外传，两者会长期并存并按素材来源切换。
    """

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[tuple[TranscriptLine, ...], str, str]:
        """返回 (句子序列, 识别出的语言, 模型标识)。

        返回元组而不是直接返回 `Transcript`，是因为 `audio_asset_id` 属于业务侧
        信息，适配器不应该知道资产中心的存在。
        """
        ...
