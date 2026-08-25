"""T1-2 验收：产物仓储契约（设计文档 11.2、9.2）。

**这套断言会跑在每一个仓储实现上。** 目前只有内存实现；PostgreSQL 实现接进来
时只需在 `REPOSITORY_FACTORIES` 里加一行，不改任何断言。这是双实现不漂移的
唯一保证——两个实现被同一套测试约束，谁少实现一条不变量谁就红。

守住的核心不变量：
1. 修改产生新版本，旧版本永远可查，且没有任何接口能改已有版本的内容；
2. 上游产生新版本时，下游被**递归**标记为可能过期，且不被覆盖；
3. 过期产物不可用于生产，也不能被直接批准；
4. 驳回是终态，翻案必须产生新版本。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.domain.errors import CapabilityError, ErrorCode
from app.domain.refs import ArtifactRef, ArtifactType
from app.domain.versioning import ArtifactRepository, ArtifactStatus
from app.storage.memory import InMemoryArtifactRepository

# 新增实现时在此登记，整套契约测试自动覆盖它。
REPOSITORY_FACTORIES: dict[str, Callable[[], ArtifactRepository]] = {
    "memory": InMemoryArtifactRepository,
}

ANALYSIS = ArtifactType.VIDEO_ANALYSIS
PATTERN = ArtifactType.SCRIPT_PATTERN
SCRIPT = ArtifactType.MARKETING_SCRIPT
STORYBOARD = ArtifactType.STORYBOARD


@pytest.fixture(params=sorted(REPOSITORY_FACTORIES), ids=sorted(REPOSITORY_FACTORIES))
def repo(request: pytest.FixtureRequest) -> Iterator[ArtifactRepository]:
    yield REPOSITORY_FACTORIES[str(request.param)]()


def make(
    repo: ArtifactRepository,
    artifact_type: ArtifactType,
    artifact_id: str,
    *,
    sources: tuple[ArtifactRef, ...] = (),
    status: ArtifactStatus = ArtifactStatus.READY,
    **body: Any,
) -> ArtifactRef:
    return repo.create_version(
        artifact_type,
        artifact_id,
        body or {"note": "x"},
        created_by="user_001",
        sources=sources,
        status=status,
    ).ref


# --------------------------------------------------------------- 版本不覆盖


def test_versions_start_at_one_and_increment(repo: ArtifactRepository) -> None:
    assert make(repo, ANALYSIS, "va_001").version == 1
    assert make(repo, ANALYSIS, "va_001").version == 2
    assert make(repo, ANALYSIS, "va_001").version == 3


def test_caller_cannot_choose_version_number(repo: ArtifactRepository) -> None:
    """版本号由仓储分配。允许调用方指定就会出现覆盖或空洞。"""
    first = make(repo, ANALYSIS, "va_001")
    second = make(repo, ANALYSIS, "va_001")
    assert (first.version, second.version) == (1, 2)


def test_old_versions_remain_queryable(repo: ArtifactRepository) -> None:
    """设计文档 11.2：新修改产生新版本，不覆盖旧版本。"""
    v1 = make(repo, ANALYSIS, "va_001", note="第一版")
    make(repo, ANALYSIS, "va_001", note="第二版")

    assert repo.get(v1).body["note"] == "第一版"
    assert repo.latest(ANALYSIS, "va_001").body["note"] == "第二版"
    assert [v.version for v in repo.history(ANALYSIS, "va_001")] == [1, 2]


def test_stored_body_is_isolated_from_caller_mutation(repo: ArtifactRepository) -> None:
    """调用方之后改传入的 dict 不应改到已落地版本，那等于绕过内容不可变。"""
    payload: dict[str, Any] = {"note": "原始"}
    ref = repo.create_version(
        ANALYSIS, "va_001", payload, created_by="user_001", status=ArtifactStatus.READY
    ).ref
    payload["note"] = "被篡改"
    assert repo.get(ref).body["note"] == "原始"


def test_repository_exposes_no_body_mutation_api(repo: ArtifactRepository) -> None:
    """协议里不存在任何能改已有版本内容的方法。

    这是结构性保证：没有接口，就不需要靠纪律。
    """
    forbidden = {"update", "update_version", "set_body", "overwrite", "delete", "put"}
    assert forbidden.isdisjoint(dir(repo))


def test_missing_version_raises_not_found(repo: ArtifactRepository) -> None:
    make(repo, ANALYSIS, "va_001")
    with pytest.raises(CapabilityError) as excinfo:
        repo.get(ArtifactRef(type=ANALYSIS, id="va_001", version=99))
    assert excinfo.value.code is ErrorCode.ARTIFACT_NOT_FOUND


def test_source_must_exist(repo: ArtifactRepository) -> None:
    """引用不存在的上游会让血缘出现断点，而断点要等到回溯时才发现。"""
    ghost = ArtifactRef(type=ANALYSIS, id="va_ghost", version=1)
    with pytest.raises(CapabilityError) as excinfo:
        make(repo, PATTERN, "sp_001", sources=(ghost,))
    assert excinfo.value.code is ErrorCode.ARTIFACT_NOT_FOUND


# --------------------------------------------------------------- 过期传播


def test_new_upstream_version_marks_direct_dependent_stale(repo: ArtifactRepository) -> None:
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))

    assert repo.get(pattern).stale is None
    make(repo, ANALYSIS, "va_001")  # 上游出 v2

    mark = repo.get(pattern).stale
    assert mark is not None
    assert mark.triggered_by.version == 2
    assert "新版本" in mark.reason


def test_staleness_propagates_transitively(repo: ArtifactRepository) -> None:
    """只标记一层等于把问题藏在第二层。

    分析 → 模板 → 脚本 → 分镜，四层都建立在旧依据上。
    """
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))
    script = make(repo, SCRIPT, "ms_001", sources=(pattern,))
    storyboard = make(repo, STORYBOARD, "sb_001", sources=(script,))

    make(repo, ANALYSIS, "va_001")  # 上游出 v2

    for ref in (pattern, script, storyboard):
        assert repo.get(ref).stale is not None, f"{ref} 未被标记为过期"


def test_staleness_does_not_overwrite_content(repo: ArtifactRepository) -> None:
    """设计文档 9.2：标记为可能过期，不直接覆盖。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,), note="原始模板")

    make(repo, ANALYSIS, "va_001")

    stored = repo.get(pattern)
    assert stored.body["note"] == "原始模板"
    assert stored.status is ArtifactStatus.READY


def test_first_stale_mark_wins(repo: ArtifactRepository) -> None:
    """原始触发源是排查起点，被后续标记覆盖后就查不到最初的原因。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))

    make(repo, ANALYSIS, "va_001")  # v2 触发首次标记
    make(repo, ANALYSIS, "va_001")  # v3 再次触发

    mark = repo.get(pattern).stale
    assert mark is not None
    assert mark.triggered_by.version == 2


def test_unrelated_artifacts_are_not_marked(repo: ArtifactRepository) -> None:
    """过度标记会让「可能过期」变成噪音，运营就不再看它了。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    make(repo, PATTERN, "sp_001", sources=(analysis_v1,))
    unrelated = make(repo, PATTERN, "sp_002")

    make(repo, ANALYSIS, "va_001")

    assert repo.get(unrelated).stale is None


def test_first_version_marks_nothing(repo: ArtifactRepository) -> None:
    analysis = make(repo, ANALYSIS, "va_001")
    assert repo.get(analysis).stale is None


def test_new_downstream_version_built_on_new_upstream_is_not_stale(
    repo: ArtifactRepository,
) -> None:
    """人工修订后产生的新版本应当是干净的，下游可以从新版本继续（设计文档 9.2）。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    make(repo, PATTERN, "sp_001", sources=(analysis_v1,))

    analysis_v2 = make(repo, ANALYSIS, "va_001")
    pattern_v2 = make(repo, PATTERN, "sp_001", sources=(analysis_v2,))

    assert repo.get(pattern_v2).stale is None


# --------------------------------------------------------------- 可用性


def test_stale_artifact_is_not_usable(repo: ArtifactRepository) -> None:
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))
    make(repo, ANALYSIS, "va_001")

    version = repo.get(pattern)
    assert version.is_usable is False
    with pytest.raises(CapabilityError) as excinfo:
        version.require_usable()
    assert excinfo.value.code is ErrorCode.ARTIFACT_VERSION_STALE


def test_draft_artifact_is_not_usable(repo: ArtifactRepository) -> None:
    ref = make(repo, ANALYSIS, "va_001", status=ArtifactStatus.DRAFT)
    assert repo.get(ref).is_usable is False


@pytest.mark.parametrize("status", [ArtifactStatus.READY, ArtifactStatus.APPROVED])
def test_ready_and_approved_are_usable(repo: ArtifactRepository, status: ArtifactStatus) -> None:
    ref = make(repo, ANALYSIS, "va_001", status=ArtifactStatus.READY)
    if status is ArtifactStatus.APPROVED:
        repo.transition(ref, ArtifactStatus.APPROVED)
    assert repo.get(ref).is_usable is True


def test_stale_artifact_cannot_be_approved(repo: ArtifactRepository) -> None:
    """依据变了就必须重新确认，不能直接盖章放行。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))
    make(repo, ANALYSIS, "va_001")

    with pytest.raises(CapabilityError) as excinfo:
        repo.transition(pattern, ArtifactStatus.APPROVED)
    assert excinfo.value.code is ErrorCode.ARTIFACT_VERSION_STALE


# --------------------------------------------------------------- 状态机


def test_draft_to_ready_to_approved(repo: ArtifactRepository) -> None:
    ref = make(repo, ANALYSIS, "va_001", status=ArtifactStatus.DRAFT)
    assert repo.transition(ref, ArtifactStatus.READY).status is ArtifactStatus.READY
    assert repo.transition(ref, ArtifactStatus.APPROVED).status is ArtifactStatus.APPROVED


def test_approved_can_still_be_rejected(repo: ArtifactRepository) -> None:
    """合规问题往往是发布后才暴露的。"""
    ref = make(repo, ANALYSIS, "va_001", status=ArtifactStatus.READY)
    repo.transition(ref, ArtifactStatus.APPROVED)
    assert repo.transition(ref, ArtifactStatus.REJECTED).status is ArtifactStatus.REJECTED


def test_rejected_is_terminal(repo: ArtifactRepository) -> None:
    """翻案必须产生新版本，否则「这一版曾被驳回」的事实会消失。"""
    ref = make(repo, ANALYSIS, "va_001", status=ArtifactStatus.READY)
    repo.transition(ref, ArtifactStatus.REJECTED)

    for target in ArtifactStatus:
        with pytest.raises(CapabilityError) as excinfo:
            repo.transition(ref, target)
        assert excinfo.value.code is ErrorCode.ARTIFACT_STATUS_TRANSITION_INVALID


def test_draft_cannot_jump_to_approved(repo: ArtifactRepository) -> None:
    """跳过 ready 等于跳过自动流程的产出，审核会失去依据。"""
    ref = make(repo, ANALYSIS, "va_001", status=ArtifactStatus.DRAFT)
    with pytest.raises(CapabilityError) as excinfo:
        repo.transition(ref, ArtifactStatus.APPROVED)
    assert excinfo.value.code is ErrorCode.ARTIFACT_STATUS_TRANSITION_INVALID


# --------------------------------------------------------------- 血缘基础


def test_dependents_returns_direct_downstream(repo: ArtifactRepository) -> None:
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    pattern_a = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))
    pattern_b = make(repo, PATTERN, "sp_002", sources=(analysis_v1,))

    assert set(repo.dependents(analysis_v1)) == {pattern_a, pattern_b}


def test_sources_record_exact_versions(repo: ArtifactRepository) -> None:
    """设计文档 3.5 要回答「某条成片由哪个分析版本产生」，逐条引用确切版本是前提。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    make(repo, ANALYSIS, "va_001")
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))

    assert repo.get(pattern).sources == (analysis_v1,)
    assert repo.get(pattern).sources[0].version == 1


def test_one_artifact_can_have_multiple_sources(repo: ArtifactRepository) -> None:
    """脚本生成同时引用模板与产品档案（设计文档 6.6）。"""
    pattern = make(repo, PATTERN, "sp_001")
    product = make(repo, ArtifactType.PRODUCT_PROFILE, "pp_001")
    script = make(repo, SCRIPT, "ms_001", sources=(pattern, product))

    assert set(repo.get(script).sources) == {pattern, product}
