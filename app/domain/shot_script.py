"""镜头脚本：把镜头、画面与台词对齐成 LLM 可分析的结构。

## 为什么需要这一层

预处理产出三样彼此独立的东西：镜头边界、关键帧、音轨转写。它们**分开时对视频
理解几乎没有价值**——模型看到一串时间区间、一堆图片、一堆句子，但不知道哪句配
哪个画面。而设计文档 6.3 要求判断「视频钩子、痛点、证据、情绪和行动号召」，
6.4 要求「还原原视频叙事结构」，这两件事都必须建立在「这个镜头画面是什么、同时
在说什么」之上。

所以在预处理与视频理解之间需要一个组织层，把三者对齐成镜头级的结构：

    镜头 i ── 时间区间
           ├─ 关键帧（画面证据）
           ├─ 画面描述（视觉理解填充，此刻为空）
           └─ 台词（落在该区间内的转写句）

## 它是计算视图，不是产物

`build_shot_script()` 是 `(镜头列表, 转写)` 的**纯函数**。刻意不落成第三个产物：

- 存下来就会出现「预处理改了、转写改了，但镜头脚本还是旧的」这种漂移，而漂移的
  症状是 LLM 分析依据了过期数据，没有任何报错；
- 它的两个上游都带确切版本号，因此视频理解产物只需引用那两个上游，血缘依然完整，
  任何时候都能按同样的纯函数重算出一模一样的视图。

## 与「分镜」的区别

设计文档 6.8 的 `Storyboard`（分镜）在**生成侧**：脚本 → 分镜 → 视频任务，它描述
「要拍成什么样」。本文件在**分析侧**：热点视频 → 镜头脚本，它描述「原视频是什么
样」。两者结构相似但方向相反，混用一个类型会让血缘图上出现语义相反的同名节点。
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.media import ShotList
from app.domain.transcript import Transcript, TranscriptLine


class ShotLine(BaseModel):
    """归属到某个镜头的一句台词。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=1)]
    overlap_ms: Annotated[int, Field(ge=0)]
    """该句落在本镜头内的时长。"""

    spans_shot_boundary: bool
    """该句是否跨越了镜头边界。

    这不是瑕疵而是重要信息：剪辑常故意让口播跨越画面切换来维持连贯，而「话没说完
    就切画面」本身就是一种创作手法。视频理解在判断叙事结构时需要看到它，否则会把
    一个连贯的表达误读成两段独立内容。
    """

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class ShotScriptEntry(BaseModel):
    """镜头脚本的一行：一个镜头的画面与台词。"""

    model_config = ConfigDict(extra="forbid")

    index: Annotated[int, Field(ge=0)]
    start_ms: Annotated[int, Field(ge=0)]
    end_ms: Annotated[int, Field(ge=1)]
    keyframe_asset_ids: tuple[str, ...] = ()
    """该镜头的所有关键帧资产 id，按时间排序。通常 3 帧（首/中/尾附近）。"""
    keyframe_at_ms: tuple[int, ...] = ()
    """与 keyframe_asset_ids 平行的时间戳。"""
    visual_description: str | None = None
    """画面描述。由视觉理解（T3-3）填充，基于该镜头全部关键帧综合而成。"""

    lines: tuple[ShotLine, ...] = ()
    speech_ms: Annotated[int, Field(ge=0)] = 0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def is_silent(self) -> bool:
        """纯画面镜头。

        在营销视频里它们承担明确功能：产品特写、字幕板、效果对比。**不能因为没有
        台词就把这个镜头从结构里省略掉**，否则视频理解会看到一个不连续的叙事。
        """
        return not self.lines

    @property
    def speech_ratio(self) -> float:
        """本镜头内有语音的时长占比。

        用于区分口播型镜头与画面型镜头——两者在叙事里的作用不同，模型需要这个信号。
        """
        return self.speech_ms / self.duration_ms if self.duration_ms else 0.0

    @property
    def text(self) -> str:
        return "".join(line.text for line in self.lines)


class ShotScript(BaseModel):
    """镜头级的画面—台词对齐视图。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entries: tuple[ShotScriptEntry, ...]
    has_transcript: bool
    """是否有转写可用。

    显式标记而不是让调用方根据「所有镜头都没台词」去猜：「无音轨」「转写失败」
    「确实全程无人说话」是三种不同情况，前两种必须能和第三种区分开，否则视频理解
    会把技术故障当成创作手法来解读。
    """

    @model_validator(mode="after")
    def _entries_are_contiguous(self) -> Self:
        for position, entry in enumerate(self.entries):
            if entry.index != position:
                raise ValueError(f"镜头脚本序号不连续：位置 {position} 的 index 为 {entry.index}")
            if position > 0 and entry.start_ms != self.entries[position - 1].end_ms:
                raise ValueError(f"镜头脚本第 {position} 行与上一行之间存在空隙或重叠")
        return self

    @property
    def shot_count(self) -> int:
        return len(self.entries)

    @property
    def total_duration_ms(self) -> int:
        return self.entries[-1].end_ms - self.entries[0].start_ms if self.entries else 0

    @property
    def silent_shot_count(self) -> int:
        return sum(1 for entry in self.entries if entry.is_silent)

    @property
    def speech_ratio(self) -> float:
        total = self.total_duration_ms
        return sum(e.speech_ms for e in self.entries) / total if total else 0.0

    @property
    def spoken_text(self) -> str:
        return "".join(entry.text for entry in self.entries)

    def unassigned_line_count(self, transcript: Transcript | None) -> int:
        """有多少转写句没能归属到任何镜头。

        正常情况下应当是 0（镜头覆盖全片）。非 0 意味着上游给的镜头列表没有覆盖
        全片，那是一个必须暴露的缺陷，不能静默丢句。
        """
        if transcript is None:
            return 0
        assigned = sum(len(entry.lines) for entry in self.entries)
        return len(transcript.lines) - assigned


def _owning_shot_index(line: TranscriptLine, shots: ShotList) -> tuple[int, int, bool]:
    """决定一句台词归属哪个镜头。

    **按时间重叠最大归属，不拆分文本。** 拆分会切断词与语义：一句「这个东西其实
    根本不用洗」被切成「这个东西其实」和「根本不用洗」，两半单独看都会被误读，而
    ASR 的词级时间戳精度不足以支撑可靠的切分点。

    返回 (镜头下标, 重叠时长, 是否跨镜头)。
    """
    best_index = -1
    best_overlap = 0
    touched = 0

    for shot in shots.shots:
        overlap = line.overlap_ms(shot.start_ms, shot.end_ms)
        if overlap <= 0:
            continue
        touched += 1
        if overlap > best_overlap:
            best_overlap, best_index = overlap, shot.index

    return best_index, best_overlap, touched > 1


def build_shot_script(
    shots: ShotList,
    transcript: Transcript | None = None,
    *,
    keyframes: tuple[tuple[str, ...], ...] = (),
    keyframe_timestamps: tuple[tuple[int, ...], ...] = (),
    frames_per_shot: int = 0,
) -> ShotScript:
    """把镜头、关键帧与转写对齐成镜头脚本。

    纯函数：同样的输入永远得到同样的输出，因此不需要作为产物存储。

    每个镜头带多帧（通常 3 帧）时，`keyframes[i]` 是第 i 镜头的 N 帧资产 id。
    """
    buckets: dict[int, list[ShotLine]] = {shot.index: [] for shot in shots.shots}
    speech: dict[int, int] = {shot.index: 0 for shot in shots.shots}

    if transcript is not None:
        for line in transcript.lines:
            index, overlap, spans = _owning_shot_index(line, shots)
            if index < 0:
                # 镜头未覆盖该时间点。ShotList 的校验器本应挡住这种情况，
                # 这里不静默丢弃，交由 unassigned_line_count() 暴露。
                continue
            buckets[index].append(
                ShotLine(
                    text=line.text,
                    start_ms=line.start_ms,
                    end_ms=line.end_ms,
                    overlap_ms=overlap,
                    spans_shot_boundary=spans,
                )
            )
            # 只累加落在本镜头内的部分，跨镜头的句子不会让占比超过 100%。
            speech[index] += overlap

    entries = tuple(
        ShotScriptEntry(
            index=shot.index,
            start_ms=shot.start_ms,
            end_ms=shot.end_ms,
            keyframe_asset_ids=(keyframes[shot.index] if shot.index < len(keyframes) else ()),
            keyframe_at_ms=(
                keyframe_timestamps[shot.index] if shot.index < len(keyframe_timestamps) else ()
            ),
            lines=tuple(buckets[shot.index]),
            speech_ms=min(speech[shot.index], shot.duration_ms),
        )
        for shot in shots.shots
    )

    return ShotScript(entries=entries, has_transcript=transcript is not None)


def render_for_analysis(script: ShotScript) -> str:
    """把镜头脚本序列化成供模型消费的紧凑文本。

    **这是数据序列化，不是提示词。** 这里只描述事实（时间、画面、台词），不含任何
    指令、角色设定或分析要求——那些属于提示词注册表（T1-5），必须先经业务评审再
    落库（计划 1.3 节）。把指令混进序列化函数会让提示词绕过评审与版本管理。
    """
    lines: list[str] = [
        f"总时长 {script.total_duration_ms}ms，共 {script.shot_count} 个镜头，"
        f"语音占比 {script.speech_ratio:.0%}" + ("" if script.has_transcript else "（无转写）")
    ]

    for entry in script.entries:
        header = f"[镜头{entry.index}] {entry.start_ms}-{entry.end_ms}ms（{entry.duration_ms}ms）"
        lines.append(header)
        lines.append(f"  画面：{entry.visual_description or '（待视觉理解）'}")
        if entry.is_silent:
            lines.append("  台词：（无）")
        else:
            for line in entry.lines:
                mark = "（跨镜头）" if line.spans_shot_boundary else ""
                lines.append(f"  台词：{line.text}{mark}")

    return "\n".join(lines)
