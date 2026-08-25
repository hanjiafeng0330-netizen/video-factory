"""资产存储的本地文件系统实现。

用于本地开发、测试和 dev 环境。生产使用对象存储实现，两者由同一套契约测试约束。

目录结构刻意与线上的桶划分一致：

    {root}/raw/{sha256 前两位}/{sha256}{扩展名}
    {root}/generated/...

按 sha256 分片是为了避免单目录下堆几十万个文件；用 sha256 而不是资产 id 做文件名
是因为去重就发生在内容层面，同内容天然共享同一个物理文件。
"""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.domain.assets import (
    AssetKind,
    AssetOrigin,
    AssetStore,
    MediaAsset,
    RightsStatus,
    SignedUrl,
    bucket_for,
    sha256_of,
)
from app.domain.errors import CapabilityError, ErrorCode


class LocalAssetStore(AssetStore):
    def __init__(self, root: Path) -> None:
        self._root = root
        self._assets: dict[str, MediaAsset] = {}
        for bucket in ("raw", "generated"):
            (root / bucket).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 写

    def put(
        self,
        path: Path,
        *,
        kind: AssetKind,
        origin: AssetOrigin,
        mime_type: str,
        created_by: str,
        rights_status: RightsStatus = RightsStatus.UNKNOWN,
        rights_note: str | None = None,
        source_url: str | None = None,
        derived_from: str | None = None,
    ) -> MediaAsset:
        if not path.is_file():
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, f"文件不存在：{path}")

        digest = sha256_of(path)
        existing = self.find_by_digest(digest, origin)
        if existing is not None:
            # 同内容同来源直接复用。这是设计文档 6.1「支持去重」的落点，也让
            # 「同一条热点被两个运营各录一次」不会产生两份文件与两条血缘。
            return existing

        bucket = bucket_for(origin)
        target = self._root / bucket.value / digest[:2] / f"{digest}{path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)

        asset = MediaAsset(
            id=f"asset_{uuid.uuid4().hex[:16]}",
            kind=kind,
            origin=origin,
            bucket=bucket,
            storage_key=str(target.relative_to(self._root)),
            mime_type=mime_type,
            size_bytes=target.stat().st_size,
            sha256=digest,
            rights_status=rights_status,
            rights_note=rights_note,
            source_url=source_url,
            derived_from=derived_from,
            created_at=datetime.now(UTC),
            created_by=created_by,
        )
        self._assets[asset.id] = asset
        return asset

    def soft_delete(self, asset_id: str, *, deleted_by: str) -> MediaAsset:
        asset = self.get(asset_id)
        if asset.deleted_at is None:
            asset.deleted_at = datetime.now(UTC)
        # 刻意不动文件。删业务记录时立即物理删除是不可逆的，而「引用它的成片
        # 其实还在」这种情况只有在删完之后才会发现。
        return asset

    # ------------------------------------------------------------------ 读

    def get(self, asset_id: str) -> MediaAsset:
        try:
            return self._assets[asset_id]
        except KeyError:
            raise CapabilityError(ErrorCode.ARTIFACT_NOT_FOUND, f"资产不存在：{asset_id}") from None

    def find_by_digest(self, sha256: str, origin: AssetOrigin) -> MediaAsset | None:
        for asset in self._assets.values():
            if asset.sha256 == sha256 and asset.origin is origin and not asset.is_deleted:
                return asset
        return None

    def open_path(self, asset_id: str) -> Path:
        asset = self.get(asset_id)
        path = self._root / asset.storage_key
        if not path.is_file():
            raise CapabilityError(
                ErrorCode.STORAGE_FAILURE,
                f"资产 {asset_id} 的元数据存在但文件缺失：{asset.storage_key}",
            )
        return path

    def signed_url(self, asset_id: str, *, ttl_seconds: int) -> SignedUrl:
        asset = self.get(asset_id)
        # 本地实现没有真正的签名机制，但仍然返回带过期时间的链接，
        # 让调用方在两种实现下走同一条代码路径。
        return SignedUrl(
            url=f"file://{self._root / asset.storage_key}",
            expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds),
        )

    def recycle_candidates(self, *, before: datetime) -> tuple[MediaAsset, ...]:
        return tuple(
            asset
            for asset in self._assets.values()
            if asset.deleted_at is not None and asset.deleted_at < before
        )
