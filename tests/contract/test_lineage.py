"""T1-3 验收：血缘由框架自动写入，全链路可回溯（设计文档 11.2、3.5）。

最重要的一条断言是 `test_no_business_code_touches_the_recorder`：整个测试文件里
没有任何一处调用 `record()`，血缘却是完整的。这正是设计目标——业务代码没有
「记得写血缘」这个义务，因为只要有一处忘写，那条链就断了。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from app.domain.lineage import LineageRepository, LineageStore, RelationType
from app.domain.refs import ArtifactRef, ArtifactType
from app.domain.versioning import ArtifactRepository, ArtifactStatus
from app.storage.memory import InMemoryArtifactRepository
from app.storage.memory_lineage import InMemoryLineage

LINEAGE_FACTORIES: dict[str, Callable[[], Any]] = {"memory": InMemoryLineage}

HOT = ArtifactType.HOT_VIDEO
ANALYSIS = ArtifactType.VIDEO_ANALYSIS
PATTERN = ArtifactType.SCRIPT_PATTERN
PRODUCT = ArtifactType.PRODUCT_PROFILE
SCRIPT = ArtifactType.MARKETING_SCRIPT
STORYBOARD = ArtifactType.STORYBOARD
JOB = ArtifactType.VIDEO_JOB
OUTPUT = ArtifactType.VIDEO_OUTPUT


@pytest.fixture(params=sorted(LINEAGE_FACTORIES), ids=sorted(LINEAGE_FACTORIES))
def lineage(request: pytest.FixtureRequest) -> Iterator[Any]:
    yield LINEAGE_FACTORIES[str(request.param)]()


@pytest.fixture
def repo(lineage: Any) -> ArtifactRepository:
    return InMemoryArtifactRepository(lineage=lineage)


def make(
    repo: ArtifactRepository,
    artifact_type: ArtifactType,
    artifact_id: str,
    *,
    sources: tuple[ArtifactRef, ...] = (),
) -> ArtifactRef:
    return repo.create_version(
        artifact_type,
        artifact_id,
        {"note": artifact_id},
        created_by="worker",
        sources=sources,
        status=ArtifactStatus.READY,
    ).ref


@pytest.fixture
def full_chain(repo: ArtifactRepository) -> dict[str, ArtifactRef]:
    """设计文档 5.1 的标准生产链路，从热点一路到成片。"""
    hot = make(repo, HOT, "hv_001")
    analysis = make(repo, ANALYSIS, "va_001", sources=(hot,))
    pattern = make(repo, PATTERN, "sp_001", sources=(analysis,))
    product = make(repo, PRODUCT, "pp_001")
    script = make(repo, SCRIPT, "ms_001", sources=(pattern, product))
    storyboard = make(repo, STORYBOARD, "sb_001", sources=(script,))
    job = make(repo, JOB, "vj_001", sources=(storyboard,))
    output = make(repo, OUTPUT, "vo_001", sources=(job,))
    return {
        "hot": hot,
        "analysis": analysis,
        "pattern": pattern,
        "product": product,
        "script": script,
        "storyboard": storyboard,
        "job": job,
        "output": output,
    }


# --------------------------------------------------------------- 自动写入


def test_no_business_code_touches_the_recorder(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    """本文件从未调用 record()，血缘却是完整的。

    这是 T1-3 的核心：血缘是框架行为，不是业务义务。
    """
    trace = lineage.trace(full_chain["output"])
    assert {node.ref for node in trace} == set(full_chain.values())


def test_relation_is_recorded_for_each_source(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    """脚本同时引用模板与产品档案，两条关系都要有。"""
    relations = lineage.relations_into(full_chain["script"])
    assert {r.source for r in relations} == {full_chain["pattern"], full_chain["product"]}


def test_relation_type_is_inferred_not_supplied(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    """关系类型由上下游类型推断。

    让调用方传的话，传错不会有任何症状——血缘看起来是完整的，只是关系名是错的，
    而这种错无法事后校验。
    """
    by_target = {
        full_chain["analysis"]: RelationType.PRODUCES,
        full_chain["pattern"]: RelationType.ABSTRACTS,
        full_chain["storyboard"]: RelationType.CONVERTED_TO,
        full_chain["job"]: RelationType.GENERATES,
    }
    for target, expected in by_target.items():
        assert lineage.relations_into(target)[0].relation_type is expected


def test_script_gets_two_different_relation_types(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    relations = {r.source: r.relation_type for r in lineage.relations_into(full_chain["script"])}
    assert relations[full_chain["pattern"]] is RelationType.GUIDES
    assert relations[full_chain["product"]] is RelationType.USED_BY


def test_recording_is_idempotent(repo: ArtifactRepository, lineage: LineageRepository) -> None:
    """能力重跑是常态（设计文档 3.1），但重跑不代表血缘变了。"""
    hot = make(repo, HOT, "hv_001")
    analysis = make(repo, ANALYSIS, "va_001", sources=(hot,))
    assert len(lineage.relations_into(analysis)) == 1


# --------------------------------------------------------------- 回溯查询


def test_trace_answers_the_design_doc_question(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    """设计文档 3.5：某条成片由哪个热点、哪个分析版本、哪个模板、哪个产品版本产生。"""
    trace = lineage.trace(full_chain["output"])
    by_type = {node.ref.type: node.ref for node in trace}

    assert by_type[HOT] == full_chain["hot"]
    assert by_type[ANALYSIS] == full_chain["analysis"]
    assert by_type[PATTERN] == full_chain["pattern"]
    assert by_type[PRODUCT] == full_chain["product"]
    # 每一层都带确切版本号，这才叫可回溯
    assert all(node.ref.version >= 1 for node in trace)


def test_trace_is_ordered_by_depth(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    trace = lineage.trace(full_chain["output"])
    depths = [node.depth for node in trace]
    assert depths == sorted(depths)
    assert trace[0].ref == full_chain["output"]
    assert trace[0].depth == 0
    assert trace[0].relation_from_child is None


def test_trace_of_root_artifact_is_itself(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    trace = lineage.trace(full_chain["hot"])
    assert len(trace) == 1
    assert trace[0].ref == full_chain["hot"]


def test_trace_does_not_include_downstream(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    """回溯是向上追溯来源，不是列出影响范围。混在一起会让报告没法读。"""
    trace = {node.ref for node in lineage.trace(full_chain["pattern"])}
    assert full_chain["script"] not in trace
    assert full_chain["output"] not in trace


def test_relations_out_of_gives_impact_scope(
    lineage: LineageRepository, full_chain: dict[str, ArtifactRef]
) -> None:
    """反向查询单独提供，用于回答「改了这个会影响谁」。"""
    out = lineage.relations_out_of(full_chain["pattern"])
    assert {r.target for r in out} == {full_chain["script"]}


def test_different_versions_are_distinct_nodes(
    repo: ArtifactRepository, lineage: LineageRepository
) -> None:
    """同一逻辑产物的两个版本在血缘里是两个节点，否则无法区分成片依据的是哪一版。"""
    analysis_v1 = make(repo, ANALYSIS, "va_001")
    analysis_v2 = make(repo, ANALYSIS, "va_001")
    pattern_a = make(repo, PATTERN, "sp_001", sources=(analysis_v1,))
    pattern_b = make(repo, PATTERN, "sp_002", sources=(analysis_v2,))

    assert lineage.relations_into(pattern_a)[0].source.version == 1
    assert lineage.relations_into(pattern_b)[0].source.version == 2


def test_workflow_run_id_is_attached_when_bound(lineage: LineageStore) -> None:
    """流程内产生的血缘要能归属到那次运行，独立调用的则为空（设计文档 5.2）。"""
    repo = InMemoryArtifactRepository(lineage=lineage)
    standalone_hot = make(repo, HOT, "hv_standalone")
    standalone = make(repo, ANALYSIS, "va_standalone", sources=(standalone_hot,))
    assert lineage.relations_into(standalone)[0].workflow_run_id is None

    repo.bind_workflow_run("wf_run_001")
    flow_hot = make(repo, HOT, "hv_flow")
    in_flow = make(repo, ANALYSIS, "va_flow", sources=(flow_hot,))
    assert lineage.relations_into(in_flow)[0].workflow_run_id == "wf_run_001"


def test_repository_without_lineage_still_works() -> None:
    """血缘是可选装配。缺了它仓储仍然可用，便于纯领域单测。"""
    repo = InMemoryArtifactRepository()
    hot = make(repo, HOT, "hv_001")
    assert make(repo, ANALYSIS, "va_001", sources=(hot,)).version == 1
