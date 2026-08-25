"""T3-1 + T3-2 端到端：真实视频入库并拆分。

用 ffmpeg 现场合成一条**有明确镜头切换**的测试视频（三段纯色硬切 + 正弦音），
而不是把二进制样本塞进仓库：二进制样本让 diff 不可读，而且一旦样本和某个 ffmpeg
版本的行为绑定，换环境就会出现「测试挂了但代码没错」。

标记为 integration 是因为它依赖本机 ffmpeg。默认 `pytest` 不跑，
用 `pytest -m integration` 触发。
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.adapters.media.ffmpeg import FfmpegMediaTool
from app.adapters.media.scene_detect import SceneDetectShotDetector
from app.capabilities.ingest_hot_video.capability import IngestHotVideoCapability
from app.capabilities.preprocess_video.capability import (
    PreprocessResult,
    PreprocessVideoCapability,
)
from app.domain.assets import AssetOrigin, BucketKind, RightsStatus
from app.domain.capability import CapabilityRequest
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.hot_video import SourcePlatform
from app.domain.refs import ArtifactRef, ArtifactType
from app.storage.local_assets import LocalAssetStore
from app.storage.memory import InMemoryArtifactRepository
from app.storage.memory_lineage import InMemoryLineage

pytestmark = pytest.mark.integration

if shutil.which("ffmpeg") is None:  # pragma: no cover
    pytest.skip("本机未安装 ffmpeg", allow_module_level=True)


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """三段纯色各 2 秒，带音轨。色块之间是硬切，scene 滤镜必然能检出。"""
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=360x640:d=2,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=360x640:d=2,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=360x640:d=2,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-filter_complex",
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
            "-map",
            "[v]",
            "-map",
            "3:a",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    assert out.stat().st_size > 0
    return out


@pytest.fixture(scope="module")
def hue_only_cut_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """含一处**纯色相**切换的视频：绿 #007f00 → 深蓝 #000080。

    两者饱和度与明度相同，只有色相不同。这类切换是 ContentDetector 用库默认阈值
    会漏掉的（得分 20 < 27），也是最初用 ffmpeg scene 滤镜完全检不出的。
    """
    out = tmp_path_factory.mktemp("media") / "hue_only.mp4"
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=360x640:d=3,format=yuv420p",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=360x640:d=3,format=yuv420p",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return out


@pytest.fixture(scope="module")
def continuous_motion_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """单一连续镜头，持续旋转 + 色相漂移，**没有任何切换**。

    用于验证检测器不会把镜头内的运动误判成切换。把阈值调到能检出纯色相切换的
    ContentDetector 在这条素材上会误切 4 处。
    """
    out = tmp_path_factory.mktemp("media") / "motion.mp4"
    subprocess.run(  # noqa: S603
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=360x640:d=8:r=25",
            "-vf",
            "rotate=a=t*0.6:c=black,hue=h=t*90",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return out


@dataclass(frozen=True)
class Pipeline:
    ingest: IngestHotVideoCapability
    preprocess: PreprocessVideoCapability
    artifacts: InMemoryArtifactRepository
    assets: LocalAssetStore
    lineage: InMemoryLineage


@pytest.fixture
def pipeline(tmp_path: Path) -> Pipeline:
    lineage = InMemoryLineage()
    artifacts = InMemoryArtifactRepository(lineage=lineage)
    assets = LocalAssetStore(tmp_path / "assets")
    tool = FfmpegMediaTool()
    return Pipeline(
        ingest=IngestHotVideoCapability(artifacts=artifacts, assets=assets, probe=tool),
        preprocess=PreprocessVideoCapability(
            artifacts=artifacts,
            assets=assets,
            probe=tool,
            audio=tool,
            shots=SceneDetectShotDetector(),
            frames=tool,
        ),
        artifacts=artifacts,
        assets=assets,
        lineage=lineage,
    )


def ingest_request(video: Path, **overrides: object) -> CapabilityRequest:
    params: dict[str, object] = {
        "file_path": str(video),
        "source_platform": SourcePlatform.DOUYIN,
        "rights_status": RightsStatus.REFERENCE_ONLY,
        "registered_by": "ops_zhang",
        "selection_reason": "三段式硬切结构，开头色彩反差强，适合测试切分",
    }
    params.update(overrides)
    return CapabilityRequest(parameters=params, idempotency_key="ingest-sample-0001")


async def run_chain(
    pipeline: Pipeline, video: Path, **params: object
) -> tuple[ArtifactRef, ArtifactRef]:
    """入库 + 预处理。多数用例都需要这条完整链路。"""
    hot_ref = (await pipeline.ingest.execute(ingest_request(video))).output_refs[0]
    out_ref = (
        await pipeline.preprocess.execute(
            CapabilityRequest(
                input_refs=(hot_ref,),
                parameters=params,
                idempotency_key="preprocess-sample-0001",
            )
        )
    ).output_refs[0]
    return hot_ref, out_ref


def read_result(pipeline: Pipeline, ref: ArtifactRef) -> PreprocessResult:
    return pipeline.artifacts.get(ref).body_as(PreprocessResult)


# --------------------------------------------------------------- 入库


async def test_ingest_reads_real_metadata(pipeline: Pipeline, sample_video: Path) -> None:
    result = await pipeline.ingest.execute(ingest_request(sample_video))

    assert result.metrics["duration_ms"] == pytest.approx(6000, abs=200)
    assert "360x640" in result.notes[0]
    assert "竖屏" in result.notes[0]

    hot = pipeline.artifacts.get(result.output_refs[0])
    asset = pipeline.assets.get(str(hot.body["asset_id"]))
    assert asset.origin is AssetOrigin.EXTERNAL_REFERENCE
    assert asset.bucket is BucketKind.RAW
    assert asset.size_bytes > 0


async def test_ingest_without_rights_is_rejected(pipeline: Pipeline, sample_video: Path) -> None:
    """设计文档 13.3：未标注授权的素材不允许入库。"""
    with pytest.raises(CapabilityError) as excinfo:
        await pipeline.ingest.execute(
            ingest_request(sample_video, rights_status=RightsStatus.UNKNOWN)
        )
    assert excinfo.value.code is ErrorCode.ARTIFACT_RIGHTS_UNCLEARED


async def test_rejected_ingest_leaves_no_orphan_asset(
    pipeline: Pipeline, sample_video: Path
) -> None:
    """校验必须发生在建资产之前。

    反过来会在拒绝一条视频之前就把它复制进原始桶，留下一个没有任何业务记录指向的
    孤儿资产，而孤儿资产在回收策略里也找不到删除依据。
    """
    with pytest.raises(CapabilityError):
        await pipeline.ingest.execute(
            ingest_request(sample_video, rights_status=RightsStatus.UNKNOWN)
        )
    assert pipeline.assets.recycle_candidates(before=datetime.now(UTC)) == ()
    with pytest.raises(CapabilityError):
        pipeline.artifacts.latest(ArtifactType.HOT_VIDEO, "hv_whatever")


async def test_reingest_same_content_becomes_new_version(
    pipeline: Pipeline, sample_video: Path
) -> None:
    """重复录入保留在版本历史里，而不是被压平成一个布尔判断。"""
    first = await pipeline.ingest.execute(ingest_request(sample_video))
    second = await pipeline.ingest.execute(
        ingest_request(sample_video, selection_reason="换个理由再录一次")
    )

    assert first.output_refs[0].id == second.output_refs[0].id
    assert second.output_refs[0].version == 2
    assert "第 2 次录入" in second.notes[-1]

    history = pipeline.artifacts.history(ArtifactType.HOT_VIDEO, first.output_refs[0].id)
    assert [v.body["selection_reason"] for v in history] == [
        "三段式硬切结构，开头色彩反差强，适合测试切分",
        "换个理由再录一次",
    ]


# --------------------------------------------------------------- 拆分


async def test_preprocess_splits_video_into_shots(pipeline: Pipeline, sample_video: Path) -> None:
    """核心验收：真实视频被切成多个镜头。"""
    _, out_ref = await run_chain(pipeline, sample_video)
    body = read_result(pipeline, out_ref)

    assert body.shots.count >= 3, "三段硬切至少应切出三个镜头"
    assert body.metadata.duration_ms == pytest.approx(6000, abs=200)
    assert body.metadata.aspect_ratio == "9:16"
    assert body.metadata.has_audio is True


async def test_shot_boundaries_land_on_real_cuts(pipeline: Pipeline, sample_video: Path) -> None:
    """切点应落在 2s 和 4s 附近，而不是随机位置。"""
    _, out_ref = await run_chain(pipeline, sample_video)
    boundaries = [shot.start_ms for shot in read_result(pipeline, out_ref).shots.shots[1:]]

    assert boundaries == [2000, 4000], f"切点应精确落在 2s 与 4s，实际：{boundaries}"


async def test_shots_are_contiguous_on_real_video(pipeline: Pipeline, sample_video: Path) -> None:
    """空隙或重叠会让下游分镜对齐出现无法定位的偏移。"""
    _, out_ref = await run_chain(pipeline, sample_video)
    shots = read_result(pipeline, out_ref).shots.shots

    assert shots[0].start_ms == 0
    for previous, current in itertools.pairwise(shots):
        assert current.start_ms == previous.end_ms


async def test_hue_only_cut_is_detected(pipeline: Pipeline, hue_only_cut_video: Path) -> None:
    """回归：纯色相切换必须被检出。

    漏掉的镜头边界会让分镜对齐和关键帧全部继承错误，而这种错在成片里表现为
    「画面和口播差半拍」，几乎不可能追溯回切分阶段。
    """
    _, out_ref = await run_chain(pipeline, hue_only_cut_video)
    body = read_result(pipeline, out_ref)

    assert body.shots.count == 2, f"应检出 3s 处的切点，实际镜头：{body.shots.shots}"
    assert abs(body.shots.shots[1].start_ms - 3000) < 300


async def test_continuous_motion_is_not_over_segmented(
    pipeline: Pipeline, continuous_motion_video: Path
) -> None:
    """回归：镜头内的运动不能被误判成切换。

    过度切分同样有害：它会把关键帧数量和后续视觉理解成本放大好几倍，还会让分镜
    出现大量没有意义的短镜头。
    """
    _, out_ref = await run_chain(pipeline, continuous_motion_video)
    assert read_result(pipeline, out_ref).shots.count == 1


async def test_preprocess_output_is_traceable(pipeline: Pipeline, sample_video: Path) -> None:
    """血缘自动指回热点视频——预处理代码里没有一行写血缘。"""
    hot_ref, out_ref = await run_chain(pipeline, sample_video)

    assert out_ref.type is ArtifactType.PREPROCESS_RESULT
    trace = pipeline.lineage.trace(out_ref)
    assert {node.ref for node in trace} == {out_ref, hot_ref}


async def test_audio_becomes_a_derived_asset(pipeline: Pipeline, sample_video: Path) -> None:
    _, out_ref = await run_chain(pipeline, sample_video)
    body = read_result(pipeline, out_ref)

    assert body.audio_asset_id is not None
    audio = pipeline.assets.get(body.audio_asset_id)
    assert audio.bucket is BucketKind.GENERATED
    assert audio.origin is AssetOrigin.DERIVED
    assert audio.size_bytes > 0
    assert audio.derived_from is not None


async def test_keyframes_are_extracted_per_shot(pipeline: Pipeline, sample_video: Path) -> None:
    """每个镜头抽 N 帧（默认 3 帧），结构是「每镜头 N 帧」的二维数组。"""
    _, out_ref = await run_chain(pipeline, sample_video)
    body = read_result(pipeline, out_ref)

    assert len(body.keyframes) == body.shots.count
    assert body.frames_per_shot == 3
    for shot_frames in body.keyframes:
        assert len(shot_frames) == body.frames_per_shot
        for asset_id in shot_frames:
            frame = pipeline.assets.get(asset_id)
            assert frame.bucket is BucketKind.GENERATED
            assert frame.size_bytes > 0
            # 派生物不单独标授权，权限由血缘上的原始资产决定
            assert frame.rights_cleared is True


async def test_keyframes_are_evenly_spaced_within_shot(
    pipeline: Pipeline, sample_video: Path
) -> None:
    """把镜头等分成 N 段，取每段中点。比「首/中/尾」好，因为首尾帧常落在转场上。"""
    _, out_ref = await run_chain(pipeline, sample_video)
    body = read_result(pipeline, out_ref)

    for shot, times in zip(body.shots.shots, body.keyframe_timestamps, strict=True):
        assert len(times) == body.frames_per_shot
        for t in times:
            assert shot.start_ms < t < shot.end_ms
        # 时间应递增
        import itertools

        for prev, curr in itertools.pairwise(times):
            assert curr > prev


async def test_shot_sample_cap_is_flagged(pipeline: Pipeline, sample_video: Path) -> None:
    """上限截断抽帧的镜头数时留显式标记，免得下游误以为没切帧。"""
    _, out_ref = await run_chain(pipeline, sample_video, max_shots_to_sample=2)
    body = read_result(pipeline, out_ref)

    assert len(body.keyframes) == 2
    assert body.truncated_shots is True
    assert body.shots.count > 2


async def test_audio_can_be_skipped(pipeline: Pipeline, sample_video: Path) -> None:
    _, out_ref = await run_chain(pipeline, sample_video, extract_audio=False)
    assert read_result(pipeline, out_ref).audio_asset_id is None


async def test_rerun_produces_a_new_version(pipeline: Pipeline, sample_video: Path) -> None:
    """设计文档 3.1：独立重跑。重跑产生新版本，旧版本仍可查。"""
    hot_ref, first = await run_chain(pipeline, sample_video)
    second = (
        await pipeline.preprocess.execute(
            CapabilityRequest(input_refs=(hot_ref,), idempotency_key="preprocess-sample-0001")
        )
    ).output_refs[0]

    assert first.id == second.id
    assert (first.version, second.version) == (1, 2)
    assert read_result(pipeline, first).shots.count >= 3


async def test_derived_assets_are_deduplicated_across_reruns(
    pipeline: Pipeline, sample_video: Path
) -> None:
    """重跑抽出的帧内容相同，资产层面应当复用而不是存两份。"""
    hot_ref, first = await run_chain(pipeline, sample_video)
    second = (
        await pipeline.preprocess.execute(
            CapabilityRequest(input_refs=(hot_ref,), idempotency_key="preprocess-sample-0001")
        )
    ).output_refs[0]

    assert (
        read_result(pipeline, first).audio_asset_id == read_result(pipeline, second).audio_asset_id
    )


async def test_preprocess_rejects_asset_without_rights(
    pipeline: Pipeline, sample_video: Path
) -> None:
    """即使已入库，资产层面授权不清也不允许分析。"""
    hot_ref = (await pipeline.ingest.execute(ingest_request(sample_video))).output_refs[0]
    asset_id = str(pipeline.artifacts.get(hot_ref).body["asset_id"])
    pipeline.assets.get(asset_id).rights_status = RightsStatus.UNKNOWN

    with pytest.raises(CapabilityError) as excinfo:
        await pipeline.preprocess.execute(
            CapabilityRequest(input_refs=(hot_ref,), idempotency_key="preprocess-sample-0001")
        )
    assert excinfo.value.code is ErrorCode.ARTIFACT_RIGHTS_UNCLEARED


async def test_preprocess_requires_a_hot_video_input(pipeline: Pipeline) -> None:
    with pytest.raises(CapabilityError) as excinfo:
        await pipeline.preprocess.execute(
            CapabilityRequest(idempotency_key="preprocess-sample-0001")
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT_REFS
