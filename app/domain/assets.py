"""资产中心（设计文档 11.1）。

设计文档的资产管理原则里，有三条会直接决定系统能不能长期活下去：

1. **业务对象只保存 `asset_id`，不复制文件。** 因此资产必须有独立身份与去重，
   否则同一条热点视频被两个运营各录一次，就会存两份文件、两条血缘。
2. **原始资产和生成资产分开管理。** 这里做得比「分开」更强一点：桶由资产来源
   `origin` 推导，调用方无法指定——否则迟早有人把 AI 生成物写进原始桶，而原始
   桶的版权口径与生成桶完全不同，混进去之后无法自动分辨。
3. **删除业务记录时不立即物理删除，先进入回收策略。** 视频文件删错是不可逆的，
   而「引用它的成片其实还在」这种情况只有在删完之后才会发现。

另外承接设计文档 13.3：外部素材必须有明确的权限口径，未标明来源与授权的素材
禁止进入生产。这条在这里是硬校验，不是提醒。
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.domain.errors import CapabilityError, ErrorCode


class AssetKind(StrEnum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"


class AssetOrigin(StrEnum):
    """资产来源。决定它落哪个桶，以及适用哪套版权口径。"""

    EXTERNAL_REFERENCE = "external_reference"
    """外部热点视频等参考素材。版权口径最严，必须显式标注授权范围。"""

    PRODUCT_MATERIAL = "product_material"
    """自有产品素材。"""

    DERIVED = "derived"
    """由系统从其他资产派生，如抽帧、抽音轨、转码。"""

    AI_GENERATED = "ai_generated"
    """AI 生成的镜头、语音、成片。"""


class BucketKind(StrEnum):
    RAW = "raw"
    GENERATED = "generated"


# 桶由 origin 推导，不接受调用方指定。
_BUCKET_BY_ORIGIN: dict[AssetOrigin, BucketKind] = {
    AssetOrigin.EXTERNAL_REFERENCE: BucketKind.RAW,
    AssetOrigin.PRODUCT_MATERIAL: BucketKind.RAW,
    AssetOrigin.DERIVED: BucketKind.GENERATED,
    AssetOrigin.AI_GENERATED: BucketKind.GENERATED,
}


class RightsStatus(StrEnum):
    """授权口径（设计文档 13.3）。"""

    UNKNOWN = "unknown"
    """未标注。禁止进入生产。"""

    REFERENCE_ONLY = "reference_only"
    """仅可用于分析与结构抽象，不可再利用素材本身。"""

    LICENSED = "licensed"
    """已获授权，可在授权范围内使用。"""

    OWNED = "owned"
    """自有素材。"""


_RIGHTS_CLEARED = frozenset(
    {RightsStatus.REFERENCE_ONLY, RightsStatus.LICENSED, RightsStatus.OWNED}
)

# 只有外部素材需要显式授权口径；派生物与 AI 生成物的权限继承自其上游，
# 由血缘决定，不在这里重复标注。
_ORIGINS_REQUIRING_RIGHTS = frozenset(
    {AssetOrigin.EXTERNAL_REFERENCE, AssetOrigin.PRODUCT_MATERIAL}
)


class MediaAsset(BaseModel):
    """资产元数据。文件本体在对象存储里，这里只描述它。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    kind: AssetKind
    origin: AssetOrigin
    bucket: BucketKind
    storage_key: str = Field(min_length=1, max_length=512)
    mime_type: str = Field(min_length=1, max_length=128)
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rights_status: RightsStatus
    rights_note: str | None = None
    source_url: str | None = None
    derived_from: str | None = None
    """派生资产的上游资产 id，如关键帧来自哪条视频。"""

    created_at: datetime
    created_by: str = Field(min_length=1, max_length=64)
    deleted_at: datetime | None = None
    """软删除时间。非空表示已进入回收策略，但文件仍在。"""

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    @property
    def rights_cleared(self) -> bool:
        if self.origin not in _ORIGINS_REQUIRING_RIGHTS:
            return True
        return self.rights_status in _RIGHTS_CLEARED

    def require_usable(self) -> None:
        """在把资产用于生产前调用。"""
        if self.is_deleted:
            raise CapabilityError(
                ErrorCode.ARTIFACT_NOT_FOUND,
                f"资产 {self.id} 已删除，不可用于生产",
            )
        if not self.rights_cleared:
            raise CapabilityError(
                ErrorCode.ARTIFACT_RIGHTS_UNCLEARED,
                f"资产 {self.id} 未标注授权口径，禁止进入生产（设计文档 13.3）",
            )


def bucket_for(origin: AssetOrigin) -> BucketKind:
    return _BUCKET_BY_ORIGIN[origin]


def sha256_of(path: Path) -> str:
    """分块计算，避免把整条视频读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class SignedUrl(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    expires_at: datetime


class AssetStore(Protocol):
    """资产存储协议。

    实现放在 `app.storage`。内存/本地文件系统实现与对象存储实现由
    `tests/contract/test_asset_store.py` 同一套契约测试约束。
    """

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
        """存入一个文件。

        实现必须保证：同内容（sha256 相同）且同 origin 的文件复用已有资产，
        不重复存储；桶由 origin 推导。
        """
        ...

    def get(self, asset_id: str) -> MediaAsset: ...

    def find_by_digest(self, sha256: str, origin: AssetOrigin) -> MediaAsset | None: ...

    def open_path(self, asset_id: str) -> Path:
        """取本地可读路径，供 ffmpeg 等本地工具使用。"""
        ...

    def signed_url(self, asset_id: str, *, ttl_seconds: int) -> SignedUrl:
        """短时有效的访问链接（设计文档 11.1）。"""
        ...

    def soft_delete(self, asset_id: str, *, deleted_by: str) -> MediaAsset:
        """标记删除并进入回收策略，不物理删除文件。"""
        ...

    def recycle_candidates(self, *, before: datetime) -> tuple[MediaAsset, ...]:
        """回收窗口已过、可以物理清理的资产。实际清理是独立运维动作。"""
        ...
