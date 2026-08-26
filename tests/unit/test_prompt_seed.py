"""草案登记：验证它们确实以 DRAFT 落地且跑不起来。

这是「先评审后落库」这条规则的最后一道守卫：即使我把提示词写进了代码，只要没有
人显式激活，它就不可能被生产链路用上。
"""

from __future__ import annotations

import pytest

from app.domain.prompt_registry import PromptRenderError, PromptStatus
from app.prompts.registry import InMemoryPromptRegistry
from app.prompts.seed import NARRATIVE_KEY, SHOT_VISUAL_KEY, seed_prompt_drafts

ALL_KEYS = (SHOT_VISUAL_KEY, NARRATIVE_KEY)


@pytest.fixture
def registry() -> InMemoryPromptRegistry:
    reg = InMemoryPromptRegistry()
    seed_prompt_drafts(reg)
    return reg


@pytest.mark.parametrize("key", ALL_KEYS)
def test_seeded_prompts_are_drafts(registry: InMemoryPromptRegistry, key: str) -> None:
    history = registry.history(key)
    assert len(history) == 1
    assert history[0].status is PromptStatus.DRAFT


@pytest.mark.parametrize("key", ALL_KEYS)
def test_seeded_prompts_cannot_run(registry: InMemoryPromptRegistry, key: str) -> None:
    """未评审的提示词不可能被生产链路用上。"""
    assert registry.requirement_report(key).ready is False
    with pytest.raises(PromptRenderError, match="没有已激活版本"):
        registry.resolve(key, {})


@pytest.mark.parametrize("key", ALL_KEYS)
def test_seeded_prompts_render_in_preview(registry: InMemoryPromptRegistry, key: str) -> None:
    """草案必须能预览，否则评审时看不到渲染后的实际形态。"""
    template = next(t for t in registry.templates() if t.key == key)
    variables: dict[str, object] = {name: f"<{name}>" for name in template.variables}
    text = registry.preview(key, 1, variables).text
    for name in template.variables:
        assert f"<{name}>" in text


def test_prompts_are_grouped_by_business_stage(
    registry: InMemoryPromptRegistry,
) -> None:
    """配置管理台按环节分组，视频理解与营销分析提示词必须各归其位。"""
    from app.prompts.seed import MARKETING_ANALYSIS_KEY

    grouped = registry.templates_by_stage()
    assert set(grouped) == {"视频理解", "营销分析"}
    assert {t.key for t in grouped["视频理解"]} == set(ALL_KEYS)
    assert {t.key for t in grouped["营销分析"]} == {MARKETING_ANALYSIS_KEY}


@pytest.mark.parametrize("key", ALL_KEYS)
def test_change_note_explains_the_draft(registry: InMemoryPromptRegistry, key: str) -> None:
    assert len(registry.get_version(key, 1).change_note) >= 8


def _flat(text: str) -> str:
    """去掉换行与缩进再匹配。

    对散文做精确匹配本来就脆：正文为了排版会在句中换行，断言不该因此失败。
    """
    return "".join(text.split())


def test_narrative_prompt_forbids_popularity_attribution() -> None:
    """设计文档 18 章：高互动不等于结构有效。

    这条约束必须写在提示词正文里，否则模型会主动给播放量归因。
    """
    from app.prompts.seed import NARRATIVE_BODY_V1

    flat = _flat(NARRATIVE_BODY_V1)
    assert "值得不值得模仿" in flat
    assert "高互动不等于结构有效" in flat


def test_visual_prompt_forbids_inventing_brands() -> None:
    """设计文档 13.3 与 18 章：不允许模型虚构画面外的品牌与产品信息。"""
    from app.prompts.seed import SHOT_VISUAL_BODY_V2

    flat = _flat(SHOT_VISUAL_BODY_V2)
    assert "不要写出帧里没有出现的品牌名、产品名、成分名" in flat
    assert "不得把台词内容当作画面内容写进描述" in flat


def test_visual_prompt_requires_composite_not_per_frame() -> None:
    """v2 禁止逐帧分述：逐帧会把一个连贯动作切成几个独立事件，下游叙事还原会误读。"""
    from app.prompts.seed import SHOT_VISUAL_BODY_V2

    flat = _flat(SHOT_VISUAL_BODY_V2)
    assert "综合描述，不逐帧分述" in flat
    assert "motion" in flat  # v2 新增的字段


def test_prompts_demand_evidence_and_separate_inference() -> None:
    """设计文档 6.3/6.4：必须区分观测事实与模型推断，且证据可追。"""
    from app.prompts.seed import NARRATIVE_BODY_V1

    flat = _flat(NARRATIVE_BODY_V1)
    assert "observed_facts" in flat
    assert "inference_notes" in flat
    assert "必须指向具体证据" in flat
    assert "区分观测与推断" in flat
