"""镜头脚本对齐逻辑。

这是 LLM 分析的输入结构，对齐错了的后果是模型「看着画面 A 读着台词 B」，而它仍会
给出一份读起来很合理的分析——这类错误不会报错，只会以「分析不准」的形式出现，
极难追溯。所以边界情况在这里测透，而不是等真实素材撞上。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.adapters.media.ffmpeg import build_shot_list
from app.domain.media import ShotList
from app.domain.shot_script import build_shot_script, render_for_analysis
from app.domain.transcript import Transcript, TranscriptLine


def shots(*cut_points: int, duration: int) -> ShotList:
    return build_shot_list(list(cut_points), duration)


def transcript(*spans: tuple[int, int, str]) -> Transcript:
    return Transcript(
        language="zh",
        model_name="test",
        audio_asset_id="asset_audio_001",
        lines=tuple(
            TranscriptLine(index=i, start_ms=s, end_ms=e, text=t)
            for i, (s, e, t) in enumerate(spans)
        ),
    )


# --------------------------------------------------------------- 基本归属


def test_lines_land_in_their_own_shots() -> None:
    script = build_shot_script(
        shots(3000, 6000, duration=9000),
        transcript((500, 2500, "第一镜"), (3200, 5500, "第二镜"), (6200, 8800, "第三镜")),
    )
    assert [entry.text for entry in script.entries] == ["第一镜", "第二镜", "第三镜"]


def test_multiple_lines_in_one_shot_keep_order() -> None:
    script = build_shot_script(
        shots(5000, duration=9000),
        transcript((200, 1000, "甲"), (1200, 2000, "乙"), (2200, 3000, "丙")),
    )
    assert script.entries[0].text == "甲乙丙"
    assert [line.text for line in script.entries[0].lines] == ["甲", "乙", "丙"]


def test_silent_shot_is_kept_not_dropped() -> None:
    """纯画面镜头在营销视频里承担明确功能：产品特写、字幕板、效果对比。

    省略它会让视频理解看到一个不连续的叙事。
    """
    script = build_shot_script(
        shots(3000, 6000, duration=9000),
        transcript((500, 2500, "只有第一镜有话")),
    )
    assert script.shot_count == 3
    assert script.entries[1].is_silent
    assert script.entries[2].is_silent
    assert script.silent_shot_count == 2


def test_no_transcript_still_produces_full_structure() -> None:
    """没有转写时结构照出，只是每个镜头都没台词。"""
    script = build_shot_script(shots(3000, duration=6000))
    assert script.shot_count == 2
    assert script.has_transcript is False
    assert all(entry.is_silent for entry in script.entries)


def test_has_transcript_distinguishes_silence_from_missing() -> None:
    """「无音轨」「转写失败」「确实无人说话」是三种不同情况。

    前两种必须能和第三种区分开，否则视频理解会把技术故障当成创作手法来解读。
    """
    without = build_shot_script(shots(duration=6000))
    with_empty = build_shot_script(shots(duration=6000), transcript())

    assert without.has_transcript is False
    assert with_empty.has_transcript is True
    assert without.entries[0].is_silent and with_empty.entries[0].is_silent


# --------------------------------------------------------------- 跨镜头台词


def test_line_spanning_boundary_goes_to_max_overlap_shot() -> None:
    """按时间重叠最大归属，不拆分文本。

    拆分会切断词与语义：「这个东西其实根本不用洗」被切成两半，两半单独看都会被
    误读，而 ASR 的词级时间戳精度不足以支撑可靠的切分点。
    """
    # 2500–4000ms：镜头0 占 500ms，镜头1 占 1000ms → 归镜头1
    script = build_shot_script(
        shots(3000, duration=9000), transcript((2500, 4000, "这句话跨了画面切换"))
    )
    assert script.entries[0].is_silent
    assert script.entries[1].text == "这句话跨了画面切换"


def test_spanning_line_is_flagged() -> None:
    """「话没说完就切画面」本身是一种创作手法，模型需要看到它。

    否则会把一个连贯的表达误读成两段独立内容。
    """
    script = build_shot_script(shots(3000, duration=9000), transcript((2500, 4000, "跨镜头")))
    assert script.entries[1].lines[0].spans_shot_boundary is True


def test_non_spanning_line_is_not_flagged() -> None:
    script = build_shot_script(shots(3000, duration=9000), transcript((500, 2500, "不跨")))
    assert script.entries[0].lines[0].spans_shot_boundary is False


def test_line_exactly_on_boundary_is_assigned_once() -> None:
    """一句话不能同时算进两个镜头，否则台词会重复出现在分析输入里。"""
    script = build_shot_script(shots(3000, duration=9000), transcript((2000, 4000, "正好一半一半")))
    assigned = [entry.text for entry in script.entries if not entry.is_silent]
    assert assigned == ["正好一半一半"]


def test_line_spanning_three_shots_lands_in_the_longest_overlap() -> None:
    # 1000–8000ms：镜头0 占 2000，镜头1 占 3000，镜头2 占 2000 → 归镜头1
    script = build_shot_script(
        shots(3000, 6000, duration=9000), transcript((1000, 8000, "很长一句"))
    )
    assert script.entries[1].text == "很长一句"
    assert script.entries[0].is_silent and script.entries[2].is_silent


# --------------------------------------------------------------- 语音占比


def test_speech_ratio_distinguishes_talking_from_visual_shots() -> None:
    """口播型镜头与画面型镜头在叙事里作用不同，模型需要这个信号。"""
    script = build_shot_script(
        shots(4000, duration=8000),
        transcript((0, 3600, "几乎说满"), (4000, 4400, "只说一点")),
    )
    assert script.entries[0].speech_ratio == pytest.approx(0.9, abs=0.01)
    assert script.entries[1].speech_ratio == pytest.approx(0.1, abs=0.01)


def test_spanning_line_does_not_inflate_speech_ratio_beyond_one() -> None:
    """只累加落在本镜头内的部分。否则跨镜头的长句会让占比超过 100%。"""
    script = build_shot_script(shots(3000, duration=9000), transcript((0, 9000, "从头说到尾")))
    assert all(entry.speech_ratio <= 1.0 for entry in script.entries)


def test_overall_speech_ratio() -> None:
    script = build_shot_script(
        shots(duration=10000), transcript((0, 2000, "甲"), (5000, 8000, "乙"))
    )
    assert script.speech_ratio == pytest.approx(0.5)


# --------------------------------------------------------------- 结构不变量


def test_entries_stay_contiguous() -> None:
    script = build_shot_script(shots(2000, 5000, 7000, duration=9000))
    assert script.entries[0].start_ms == 0
    assert script.entries[-1].end_ms == 9000
    for previous, current in zip(script.entries, script.entries[1:], strict=False):
        assert current.start_ms == previous.end_ms


def test_keyframes_are_attached_per_shot() -> None:
    """每镜头多帧（3 帧）都完整挂在对应镜头上。"""
    script = build_shot_script(
        shots(3000, duration=6000),
        keyframes=(("asset_a1", "asset_a2", "asset_a3"), ("asset_b1", "asset_b2", "asset_b3")),
        keyframe_timestamps=((750, 1500, 2250), (3750, 4500, 5250)),
    )
    assert script.entries[0].keyframe_asset_ids == ("asset_a1", "asset_a2", "asset_a3")
    assert script.entries[1].keyframe_asset_ids == ("asset_b1", "asset_b2", "asset_b3")
    assert script.entries[0].keyframe_at_ms == (750, 1500, 2250)
    assert script.entries[1].keyframe_at_ms == (3750, 4500, 5250)


def test_missing_keyframes_leave_empty_rather_than_shifting() -> None:
    """关键帧被截断时后面的镜头留空，而不是把帧错位对到别的镜头上。

    错位会让模型看着 A 的画面读 B 的台词，而它仍会给出一份读起来合理的分析。
    """
    script = build_shot_script(
        shots(2000, 4000, duration=6000),
        keyframes=(("asset_a1", "asset_a2", "asset_a3"),),
        keyframe_timestamps=((750, 1500, 2250),),
    )
    assert script.entries[0].keyframe_asset_ids == ("asset_a1", "asset_a2", "asset_a3")
    assert script.entries[1].keyframe_asset_ids == ()
    assert script.entries[2].keyframe_asset_ids == ()


def test_no_line_is_silently_dropped() -> None:
    script = build_shot_script(
        shots(3000, duration=9000),
        t := transcript((500, 1000, "甲"), (4000, 5000, "乙"), (8000, 8500, "丙")),
    )
    assert script.unassigned_line_count(t) == 0
    assert sum(len(e.lines) for e in script.entries) == 3


def test_transcript_rejects_unordered_lines() -> None:
    """乱序会让按时间归属镜头的逻辑静默出错，所以在入口就拒掉。"""
    with pytest.raises(ValidationError, match="未按时间排序"):
        transcript((7000, 8000, "丙"), (500, 1000, "甲"))


def test_spoken_text_follows_timeline_order() -> None:
    script = build_shot_script(
        shots(3000, 6000, duration=9000),
        transcript((500, 1000, "甲"), (4000, 5000, "乙"), (7000, 8000, "丙")),
    )
    assert script.spoken_text == "甲乙丙"


# --------------------------------------------------------------- 序列化


def test_render_contains_shots_visuals_and_lines() -> None:
    script = build_shot_script(shots(3000, duration=6000), transcript((500, 2500, "开场一句")))
    text = render_for_analysis(script)

    assert "[镜头0]" in text and "[镜头1]" in text
    assert "开场一句" in text
    assert "画面：" in text
    assert "台词：（无）" in text  # 第二个镜头无台词


def test_render_flags_missing_transcript() -> None:
    assert "（无转写）" in render_for_analysis(build_shot_script(shots(duration=6000)))


def test_render_marks_spanning_lines() -> None:
    script = build_shot_script(shots(3000, duration=9000), transcript((2500, 4000, "跨")))
    assert "（跨镜头）" in render_for_analysis(script)


def test_render_carries_no_instructions() -> None:
    """这是数据序列化，不是提示词。

    指令、角色设定、分析要求属于提示词注册表（T1-5），必须先经业务评审再落库。
    混进序列化函数会让提示词绕过评审与版本管理。
    """
    text = render_for_analysis(
        build_shot_script(shots(3000, duration=6000), transcript((500, 2500, "台词")))
    )
    for instruction_word in ("请", "你是", "分析", "输出", "要求", "任务"):
        assert instruction_word not in text
