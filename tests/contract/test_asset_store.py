"""T1-1 验收：资产中心契约（设计文档 11.1、13.3）。

同一套断言跑在每个资产存储实现上。对象存储实现接入时只需在
`ASSET_STORE_FACTORIES` 加一行。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.domain.assets import (
    AssetKind,
    AssetOrigin,
    AssetStore,
    BucketKind,
    MediaAsset,
    RightsStatus,
    sha256_of,
)
from app.domain.errors import CapabilityError, ErrorCode
from app.storage.local_assets import LocalAssetStore

ASSET_STORE_FACTORIES: dict[str, Callable[[Path], AssetStore]] = {
    "local": LocalAssetStore,
}


@pytest.fixture(params=sorted(ASSET_STORE_FACTORIES), ids=sorted(ASSET_STORE_FACTORIES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[AssetStore]:
    root = tmp_path / "store"
    root.mkdir()
    yield ASSET_STORE_FACTORIES[str(request.param)](root)


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake-video-bytes")
    return path


def put_external(
    store: AssetStore, path: Path, *, rights: RightsStatus = RightsStatus.REFERENCE_ONLY
) -> MediaAsset:
    return store.put(
        path,
        kind=AssetKind.VIDEO,
        origin=AssetOrigin.EXTERNAL_REFERENCE,
        mime_type="video/mp4",
        created_by="user_001",
        rights_status=rights,
    )


# --------------------------------------------------------------- 去重


def test_same_content_is_deduplicated(store: AssetStore, video_file: Path) -> None:
    """同一条热点被两个运营各录一次，不应产生两份文件与两条血缘。"""
    first = put_external(store, video_file)
    second = put_external(store, video_file)
    assert first == second


def test_different_content_creates_distinct_assets(store: AssetStore, tmp_path: Path) -> None:
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"aaa")
    b.write_bytes(b"bbb")
    assert put_external(store, a) != put_external(store, b)


def test_digest_is_recorded_and_correct(store: AssetStore, video_file: Path) -> None:
    asset = store.put(
        video_file,
        kind=AssetKind.VIDEO,
        origin=AssetOrigin.EXTERNAL_REFERENCE,
        mime_type="video/mp4",
        created_by="user_001",
        rights_status=RightsStatus.REFERENCE_ONLY,
    )
    assert asset.sha256 == sha256_of(video_file)


def test_same_content_different_origin_is_not_shared(store: AssetStore, video_file: Path) -> None:
    """外部参考素材与自有素材的版权口径不同，不能因为字节相同就合并成一条。"""
    external = store.put(
        video_file,
        kind=AssetKind.VIDEO,
        origin=AssetOrigin.EXTERNAL_REFERENCE,
        mime_type="video/mp4",
        created_by="user_001",
        rights_status=RightsStatus.REFERENCE_ONLY,
    )
    owned = store.put(
        video_file,
        kind=AssetKind.VIDEO,
        origin=AssetOrigin.PRODUCT_MATERIAL,
        mime_type="video/mp4",
        created_by="user_001",
        rights_status=RightsStatus.OWNED,
    )
    assert external.id != owned.id
    assert external.bucket is owned.bucket is BucketKind.RAW


# --------------------------------------------------------------- 分桶


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        (AssetOrigin.EXTERNAL_REFERENCE, BucketKind.RAW),
        (AssetOrigin.PRODUCT_MATERIAL, BucketKind.RAW),
        (AssetOrigin.DERIVED, BucketKind.GENERATED),
        (AssetOrigin.AI_GENERATED, BucketKind.GENERATED),
    ],
)
def test_bucket_is_derived_from_origin(
    store: AssetStore, video_file: Path, origin: AssetOrigin, expected: BucketKind
) -> None:
    """桶由来源推导，调用方无法指定。

    否则迟早有人把 AI 生成物写进原始桶，而两个桶的版权口径完全不同，
    混进去之后无法自动分辨。
    """
    asset = store.put(
        video_file,
        kind=AssetKind.VIDEO,
        origin=origin,
        mime_type="video/mp4",
        created_by="user_001",
        rights_status=RightsStatus.OWNED,
    )
    assert asset.bucket is expected


def test_store_accepts_no_bucket_argument(store: AssetStore) -> None:
    """结构性保证：接口里没有 bucket 参数，就不可能传错。"""
    import inspect

    assert "bucket" not in inspect.signature(store.put).parameters


# --------------------------------------------------------------- 权限口径


def test_external_asset_without_rights_is_unusable(store: AssetStore, video_file: Path) -> None:
    """设计文档 13.3：未标明授权的外部素材禁止进入生产。"""
    asset = store.put(
        video_file,
        kind=AssetKind.VIDEO,
        origin=AssetOrigin.EXTERNAL_REFERENCE,
        mime_type="video/mp4",
        created_by="user_001",
    )
    assert asset.rights_cleared is False
    with pytest.raises(CapabilityError) as excinfo:
        asset.require_usable()
    assert excinfo.value.code is ErrorCode.ARTIFACT_RIGHTS_UNCLEARED


@pytest.mark.parametrize(
    "rights",
    [RightsStatus.REFERENCE_ONLY, RightsStatus.LICENSED, RightsStatus.OWNED],
)
def test_external_asset_with_rights_is_usable(
    store: AssetStore, video_file: Path, rights: RightsStatus
) -> None:
    asset = store.put(
        video_file,
        kind=AssetKind.VIDEO,
        origin=AssetOrigin.EXTERNAL_REFERENCE,
        mime_type="video/mp4",
        created_by="user_001",
        rights_status=rights,
    )
    asset.require_usable()


def test_derived_asset_inherits_rights_from_lineage(store: AssetStore, video_file: Path) -> None:
    """派生物不重复标注授权，其权限由血缘上的原始资产决定。"""
    frame = store.put(
        video_file,
        kind=AssetKind.IMAGE,
        origin=AssetOrigin.DERIVED,
        mime_type="image/jpeg",
        created_by="worker",
        derived_from="asset_source_001",
    )
    assert frame.rights_status is RightsStatus.UNKNOWN
    assert frame.rights_cleared is True
    frame.require_usable()
    assert frame.derived_from == "asset_source_001"


# --------------------------------------------------------------- 生命周期


def test_soft_delete_keeps_the_file(store: AssetStore, video_file: Path) -> None:
    """设计文档 11.1：删除业务记录时不立即物理删除，先进入回收策略。"""
    asset = put_external(store, video_file)
    deleted = store.soft_delete(asset.id, deleted_by="user_001")

    assert deleted.is_deleted
    assert store.open_path(deleted.id).is_file()


def test_deleted_asset_is_unusable(store: AssetStore, video_file: Path) -> None:
    asset = put_external(store, video_file)
    deleted = store.soft_delete(asset.id, deleted_by="user_001")
    with pytest.raises(CapabilityError):
        deleted.require_usable()


def test_soft_delete_is_idempotent(store: AssetStore, video_file: Path) -> None:
    """重复删除不应刷新回收计时，否则资产永远等不到回收窗口。"""
    asset = put_external(store, video_file)
    first = store.soft_delete(asset.id, deleted_by="user_001")
    stamp = first.deleted_at
    second = store.soft_delete(asset.id, deleted_by="user_001")
    assert second.deleted_at == stamp


def test_recycle_candidates_only_include_expired_deletions(
    store: AssetStore, video_file: Path
) -> None:
    asset = put_external(store, video_file)
    assert store.recycle_candidates(before=datetime.now(UTC)) == ()

    store.soft_delete(asset.id, deleted_by="user_001")
    assert store.recycle_candidates(before=datetime.now(UTC) - timedelta(days=30)) == ()
    assert len(store.recycle_candidates(before=datetime.now(UTC) + timedelta(seconds=1))) == 1


def test_missing_asset_raises_not_found(store: AssetStore) -> None:
    with pytest.raises(CapabilityError) as excinfo:
        store.get("asset_nope")
    assert excinfo.value.code is ErrorCode.ARTIFACT_NOT_FOUND


# --------------------------------------------------------------- 访问链接


def test_signed_url_carries_expiry(store: AssetStore, video_file: Path) -> None:
    """设计文档 11.1：访问地址使用短时有效的签名链接。"""
    asset = put_external(store, video_file)
    signed = store.signed_url(asset.id, ttl_seconds=300)
    delta = signed.expires_at - datetime.now(UTC)
    assert timedelta(seconds=240) < delta <= timedelta(seconds=300)


def test_put_rejects_missing_file(store: AssetStore, tmp_path: Path) -> None:
    with pytest.raises(CapabilityError) as excinfo:
        put_external(store, tmp_path / "nope.mp4")
    assert excinfo.value.code is ErrorCode.INVALID_PARAMETERS
