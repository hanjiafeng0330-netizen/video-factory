"""切点 → 镜头列表的纯逻辑。

单独测这段是因为这些边界情况用真实视频去构造既慢又难：切点重复、切点落在 0ms、
切点越界、末尾残留几十毫秒。它们都是实际会遇到的——scene 滤镜在快剪片段上会
连续吐出好几个相邻切点。
"""

from __future__ import annotations

import pytest

from app.adapters.media.ffmpeg import build_shot_list
from app.domain.errors import CapabilityError


def test_no_cut_points_yields_one_full_length_shot() -> None:
    """检测不到切换时返回单个覆盖全片的镜头，而不是空列表。

    空列表会让下游到处判空。
    """
    shots = build_shot_list([], 5000)
    assert shots.count == 1
    assert (shots.shots[0].start_ms, shots.shots[0].end_ms) == (0, 5000)


def test_cut_points_become_boundaries() -> None:
    shots = build_shot_list([2000, 4000], 6000)
    assert [(s.start_ms, s.end_ms) for s in shots.shots] == [
        (0, 2000),
        (2000, 4000),
        (4000, 6000),
    ]


def test_shots_are_contiguous_and_cover_whole_video() -> None:
    """允许空隙或重叠会让下游分镜对齐出现无法定位的偏移。"""
    shots = build_shot_list([1500, 3200, 4800], 6000)
    assert shots.shots[0].start_ms == 0
    assert shots.shots[-1].end_ms == 6000
    assert shots.total_duration_ms == 6000


def test_adjacent_cut_points_are_collapsed() -> None:
    """快剪闪切会连续吐出相邻切点，那些「镜头」抽帧没有分析价值。"""
    shots = build_shot_list([2000, 2050, 2100, 2150], 6000)
    assert shots.count == 2


def test_zero_and_out_of_range_cut_points_are_ignored() -> None:
    shots = build_shot_list([0, -100, 6000, 99999, 3000], 6000)
    assert [(s.start_ms, s.end_ms) for s in shots.shots] == [(0, 3000), (3000, 6000)]


def test_trailing_sliver_is_merged_into_previous_shot() -> None:
    """末尾残片并回上一镜头，而不是留一个几十毫秒的镜头。"""
    shots = build_shot_list([3000, 5950], 6000)
    assert [(s.start_ms, s.end_ms) for s in shots.shots] == [(0, 3000), (3000, 6000)]


def test_single_short_video_still_yields_one_shot() -> None:
    """整片本身短于最小镜头长度时也必须有一个镜头。"""
    shots = build_shot_list([], 200)
    assert shots.count == 1
    assert shots.shots[0].end_ms == 200


def test_zero_duration_is_rejected() -> None:
    with pytest.raises(CapabilityError, match="时长为 0"):
        build_shot_list([], 0)


def test_shot_indexes_are_sequential() -> None:
    shots = build_shot_list([1000, 2000, 3000, 4000], 5000)
    assert [s.index for s in shots.shots] == [0, 1, 2, 3, 4]


def test_midpoint_is_used_for_keyframes() -> None:
    """首帧常落在转场上，抽出来是黑帧或叠化的糊图。"""
    shots = build_shot_list([2000], 6000)
    assert shots.shots[0].midpoint_ms == 1000
    assert shots.shots[1].midpoint_ms == 4000


# --------------------------------------------------------------- 灵敏度映射


@pytest.mark.parametrize(
    ("sensitivity", "expected"),
    [(0.0, 5.0), (0.5, 3.0), (1.0, 1.0)],
)
def test_sensitivity_maps_to_library_threshold(sensitivity: float, expected: float) -> None:
    """灵敏度高 → 阈值低。方向搞反的症状是「镜头数不对」，很容易被误当成视频特性。"""
    from app.adapters.media.scene_detect import sensitivity_to_threshold

    assert sensitivity_to_threshold(sensitivity) == pytest.approx(expected)


def test_default_sensitivity_hits_the_library_default() -> None:
    """默认值必须落在已知可用的点上。

    第一版区间定成 12~45、领域默认取 0.3，映射结果比库自身默认值还不敏感，
    实测漏掉了同饱和度同明度、仅色相不同的切换。
    """
    from app.adapters.media.scene_detect import sensitivity_to_threshold
    from app.capabilities.preprocess_video.capability import PreprocessParameters

    default = PreprocessParameters()
    assert sensitivity_to_threshold(default.shot_sensitivity) == pytest.approx(3.0)


def test_sensitivity_is_clamped() -> None:
    from app.adapters.media.scene_detect import sensitivity_to_threshold

    assert sensitivity_to_threshold(-5.0) == pytest.approx(5.0)
    assert sensitivity_to_threshold(9.0) == pytest.approx(1.0)
