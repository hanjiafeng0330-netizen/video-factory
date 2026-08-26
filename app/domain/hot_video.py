"""热点视频（设计文档 6.1、7.2）。

一期只支持手动上传与链接录入元信息，不做自动采集（开发计划第 3 章默认假设）。

两条规则值得单独说明。

**授权口径在入库时必填，不是使用时再检查。** 入库时运营手上有上下文——他刚从
哪里看到这条视频；到了生成脚本那一步已经没人说得清素材是哪来的了。所以校验
放在最前面（设计文档 13.3）。

**入库理由必填，且不接受一两个字的敷衍。** 设计文档 18 章要求「保留人工判断，
结合内容结构和人群，而非只看播放量」。把它做成必填是让这条要求在流程里真的
发生，而不是停留在文档上。配套地，`metrics_snapshot` 带 `captured_at`——指标
随时间变化，不带采集时间的数字在三个月后没有意义，还会被误当成当前值参与判断。

去重不靠额外字段：逻辑 id 由内容摘要推导，同一条视频被再次录入时自然成为同一
逻辑产物的新版本。这样「谁在什么时候又录了一次、写了什么理由」全部保留在版本
历史里，而不是被一个 `duplicate_of` 布尔判断压平。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.domain.assets import RightsStatus


class SourcePlatform(StrEnum):
    DOUYIN = "douyin"
    KUAISHOU = "kuaishou"
    XIAOHONGSHU = "xiaohongshu"
    BILIBILI = "bilibili"
    WECHAT_CHANNELS = "wechat_channels"
    OTHER = "other"


class MetricsSnapshot(BaseModel):
    """互动指标快照，不是当前值。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    captured_at: datetime
    views: Annotated[int, Field(ge=0)] = 0
    likes: Annotated[int, Field(ge=0)] = 0
    comments: Annotated[int, Field(ge=0)] = 0
    shares: Annotated[int, Field(ge=0)] = 0


class HotVideoBody(BaseModel):
    """热点视频产物的 body。对应设计文档 7.2 的 `HotVideo`。"""

    model_config = ConfigDict(extra="forbid")

    source_platform: SourcePlatform
    source_url: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    asset_id: str = Field(min_length=1, max_length=64)
    original_filename: str = Field(default="未记录文件名", min_length=1, max_length=255)
    metrics_snapshot: MetricsSnapshot | None = None
    tags: tuple[str, ...] = ()
    rights_status: RightsStatus
    registered_by: str = Field(min_length=1, max_length=64)
    selection_reason: str = Field(min_length=4, max_length=500)
    """为什么把它选进来。运营的判断，不是指标的判断。"""


def logical_id_for(sha256: str) -> str:
    """由内容摘要推导逻辑 id。

    同内容的视频落到同一个逻辑产物上，重复录入即新版本。用摘要而非自增 id 是为了
    让「这条视频是否已经入过库」成为一个纯函数问题，不需要先查库再决定怎么写。
    """
    return f"hv_{sha256[:16]}"
