"""T1-5 验收：提示词注册表（开发计划 1.3 节）。

最核心的一条是 `test_draft_cannot_be_used_in_production`：它把「提示词写完先给业务
评审」从口头约定变成技术事实——开发把草案写进注册表这个动作是安全的，因为草案
跑不起来。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.domain.idempotency import AuditAction
from app.domain.prompt_registry import PromptRenderError, PromptStatus, PromptTemplate
from app.prompts.registry import InMemoryPromptRegistry
from app.storage.memory_governance import InMemoryAuditLog

REGISTRY_FACTORIES: dict[str, Callable[[Any], Any]] = {"memory": InMemoryPromptRegistry}

KEY = "video_understand.narrative"

TEMPLATE = PromptTemplate(
    key=KEY,
    stage="analysis",
    purpose="基于镜头脚本判断叙事结构",
    variables=("shot_script", "platform"),
)


@pytest.fixture
def audit() -> InMemoryAuditLog:
    return InMemoryAuditLog()


@pytest.fixture(params=sorted(REGISTRY_FACTORIES), ids=sorted(REGISTRY_FACTORIES))
def registry(request: pytest.FixtureRequest, audit: InMemoryAuditLog) -> Iterator[Any]:
    reg = REGISTRY_FACTORIES[str(request.param)](audit)
    reg.register(TEMPLATE)
    yield reg


def add(registry: Any, body: str = "分析 {{ shot_script }}，平台 {{ platform }}") -> Any:
    return registry.add_version(KEY, body, change_note="初版草案", author="dev_a")


# --------------------------------------------------------------- 评审门禁


def test_new_version_lands_as_draft(registry: Any) -> None:
    assert add(registry).status is PromptStatus.DRAFT


def test_draft_cannot_be_used_in_production(registry: Any) -> None:
    """核心断言：草案跑不起来。

    这让「先评审后落库」成为技术事实而不是纪律——开发写草案入库是安全动作。
    """
    add(registry)
    with pytest.raises(PromptRenderError, match="没有已激活版本"):
        registry.resolve(KEY, {"shot_script": "x", "platform": "douyin"})


def test_draft_error_message_points_to_the_review_step(registry: Any) -> None:
    """报错要能自解释，否则运营看到的是一个无从下手的异常。"""
    add(registry)
    with pytest.raises(PromptRenderError, match=r"草案 \[1\].*业务评审"):
        registry.resolve(KEY, {"shot_script": "x", "platform": "douyin"})


def test_resolve_version_also_refuses_drafts(registry: Any) -> None:
    """指定版本号也不能绕过门禁。"""
    add(registry)
    with pytest.raises(PromptRenderError, match="不可用于生产"):
        registry.resolve_version(KEY, 1, {"shot_script": "x", "platform": "douyin"})


def test_preview_can_render_drafts(registry: Any) -> None:
    """预览是给人看的，解析是给生产用的。

    分开命名是刻意的：混成一个函数迟早会有人用预览结果去跑生产。
    """
    add(registry)
    preview = registry.preview(KEY, 1, {"shot_script": "镜头脚本", "platform": "douyin"})
    assert "镜头脚本" in preview.text


def test_activation_makes_it_usable(registry: Any) -> None:
    add(registry)
    registry.activate(KEY, 1, actor="reviewer_zhang")
    resolved = registry.resolve(KEY, {"shot_script": "镜头脚本", "platform": "douyin"})
    assert resolved.version == 1
    assert "镜头脚本" in resolved.text


def test_activation_is_audited(registry: Any, audit: InMemoryAuditLog) -> None:
    """谁在什么时候激活了哪一版，必须留痕。"""
    add(registry)
    registry.activate(KEY, 1, actor="reviewer_zhang")

    entries = audit.entries_for(f"prompt:{KEY}")
    assert len(entries) == 1
    assert entries[0].action is AuditAction.PROMPT_CHANGED
    assert entries[0].actor == "reviewer_zhang"
    assert "v1" in entries[0].summary


# --------------------------------------------------------------- 版本


def test_versions_increment_and_body_is_immutable(registry: Any) -> None:
    """改动即新版本，旧版本正文永远可渲染。"""
    add(registry, "第一版 {{ shot_script }} {{ platform }}")
    add(registry, "第二版 {{ shot_script }} {{ platform }}")

    assert [v.version for v in registry.history(KEY)] == [1, 2]
    assert "第一版" in registry.get_version(KEY, 1).body
    assert "第二版" in registry.get_version(KEY, 2).body


def test_activating_new_version_retires_the_old_one(registry: Any) -> None:
    add(registry, "第一版 {{ shot_script }} {{ platform }}")
    add(registry, "第二版 {{ shot_script }} {{ platform }}")
    registry.activate(KEY, 1, actor="reviewer")
    registry.activate(KEY, 2, actor="reviewer")

    assert registry.get_version(KEY, 1).status is PromptStatus.RETIRED
    assert registry.get_version(KEY, 2).status is PromptStatus.ACTIVE
    assert registry.active_version(KEY).version == 2


def test_retired_version_still_renders(registry: Any) -> None:
    """历史产物必须能用当初那一版重跑。"""
    add(registry, "第一版 {{ shot_script }} {{ platform }}")
    add(registry, "第二版 {{ shot_script }} {{ platform }}")
    registry.activate(KEY, 1, actor="reviewer")
    registry.activate(KEY, 2, actor="reviewer")

    old = registry.resolve_version(KEY, 1, {"shot_script": "x", "platform": "douyin"})
    assert "第一版" in old.text


def test_rollback_by_activating_an_older_version(registry: Any) -> None:
    add(registry, "第一版 {{ shot_script }} {{ platform }}")
    add(registry, "第二版 {{ shot_script }} {{ platform }}")
    registry.activate(KEY, 2, actor="reviewer")
    registry.activate(KEY, 1, actor="reviewer")

    assert registry.active_version(KEY).version == 1
    assert registry.get_version(KEY, 2).status is PromptStatus.RETIRED


def test_change_note_is_mandatory(registry: Any) -> None:
    """「上一版为什么不够用」是后续调优唯一可靠的线索。"""
    with pytest.raises(Exception, match=r"change_note|string_too_short"):
        registry.add_version(
            KEY, "正文 {{ shot_script }} {{ platform }}", change_note="", author="dev_a"
        )


# --------------------------------------------------------------- 变量契约


def test_body_cannot_use_undeclared_variables(registry: Any) -> None:
    """在写入时校验，而不是等渲染时才发现。

    草案入库和激活之间可能隔很久，那时作者已经不在上下文里了。
    """
    with pytest.raises(PromptRenderError, match="未声明的变量"):
        registry.add_version(KEY, "用了 {{ mystery }}", change_note="x", author="dev_a")


def test_render_refuses_missing_variables(registry: Any) -> None:
    """缺变量必须报错，不能渲染出一个带空洞的提示词。

    模型收到缺了关键上下文的提示词仍会给出看起来合理的输出，而这种错没有症状。
    """
    add(registry)
    registry.activate(KEY, 1, actor="reviewer")
    with pytest.raises(PromptRenderError, match="缺少变量"):
        registry.resolve(KEY, {"shot_script": "x"})


def test_render_substitutes_all_variables(registry: Any) -> None:
    add(registry)
    registry.activate(KEY, 1, actor="reviewer")
    text = registry.resolve(KEY, {"shot_script": "SS", "platform": "PP"}).text
    assert "SS" in text and "PP" in text
    assert "{{" not in text


def test_template_syntax_error_is_caught_at_write_time(registry: Any) -> None:
    with pytest.raises(PromptRenderError, match="语法错误"):
        registry.add_version(KEY, "{% if %}坏模板", change_note="x", author="dev_a")


def test_unregistered_key_is_rejected(registry: Any) -> None:
    with pytest.raises(PromptRenderError, match="未登记"):
        registry.add_version("nope.key", "x", change_note="y", author="dev_a")


def test_re_registering_with_different_contract_is_rejected(registry: Any) -> None:
    """契约变了意味着调用方的注入代码也要改，静默覆盖会让两边错位。"""
    with pytest.raises(PromptRenderError, match="契约不同"):
        registry.register(
            PromptTemplate(key=KEY, stage="analysis", purpose="改了变量", variables=("only_one",))
        )


# --------------------------------------------------------------- 后台展示


def test_templates_grouped_by_stage(registry: Any) -> None:
    """配置管理台（T2-6）按环节分组展示各环节用了哪些提示词。"""
    registry.register(
        PromptTemplate(
            key="script_gen.system", stage="creation", purpose="生成营销脚本", variables=()
        )
    )
    grouped = registry.templates_by_stage()
    assert set(grouped) == {"analysis", "creation"}
    assert grouped["analysis"][0].key == KEY


def test_requirement_report_tells_whether_a_stage_can_run(registry: Any) -> None:
    """「提示词还是草案」需要一个能展示给运营的答案，而不是一个异常。"""
    unregistered = registry.requirement_report("nope.key")
    assert unregistered.registered is False and unregistered.ready is False

    add(registry)
    drafted = registry.requirement_report(KEY)
    assert drafted.registered is True
    assert drafted.ready is False
    assert drafted.draft_versions == (1,)

    registry.activate(KEY, 1, actor="reviewer")
    ready = registry.requirement_report(KEY)
    assert ready.ready is True
    assert ready.active_version == 1
