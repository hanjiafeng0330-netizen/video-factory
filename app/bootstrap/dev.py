"""组合根：开发档装配。

这是整个进程里**唯一**知道「内存仓储」「本地文件系统资产」「ffmpeg」这些具体
选择的地方。其余代码只认 domain 协议，所以换成 PostgreSQL 仓储或对象存储时只改
这一层。

组合根独立成包（而不是放在 `app.api` 里）是被边界契约逼出来的，而契约是对的：
组合根必须能 import 一切，而 API 层必须什么实现都不认识。把两件事放进同一个包，
等于给「某个路由直接调适配器」开了门。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from app.adapters.asr.faster_whisper import FasterWhisperRecognizer
from app.adapters.llm.anthropic import AnthropicClient
from app.adapters.media.ffmpeg import FfmpegMediaTool
from app.adapters.media.scene_detect import SceneDetectShotDetector
from app.capabilities.registry import CapabilityDeps, CapabilityRegistry, build_capabilities
from app.platform.config import Settings, load_settings
from app.prompts.registry import InMemoryPromptRegistry
from app.prompts.seed import seed_and_activate_prompts
from app.storage.local_assets import LocalAssetStore
from app.storage.memory import InMemoryArtifactRepository
from app.storage.memory_governance import InMemoryAuditLog, InMemoryIdempotencyStore
from app.storage.memory_lineage import InMemoryLineage


@dataclass(frozen=True)
class DevContainer:
    """满足 `app.api.deps.Container` 协议。字段类型是具体实现，因为这里就是组合根。"""

    settings: Settings
    capabilities: CapabilityRegistry
    artifacts: InMemoryArtifactRepository
    assets: LocalAssetStore
    lineage: InMemoryLineage
    idempotency: InMemoryIdempotencyStore
    audit: InMemoryAuditLog
    prompts: InMemoryPromptRegistry


def build_dev_container(settings: Settings, asset_root: Path) -> DevContainer:
    if settings.environment.is_production:
        # 内存仓储重启即丢全部产物。默默用它跑生产的后果是「上周那条成片的依据
        # 查不到了」，而那时已经无法补回。
        raise RuntimeError(
            "开发档使用内存仓储，禁止用于生产环境。"
            "生产需要 PostgreSQL 与对象存储实现（计划 T1-1/T1-2 的持久化侧尚未落地）。"
        )

    lineage = InMemoryLineage()
    artifacts = InMemoryArtifactRepository(lineage=lineage)
    assets = LocalAssetStore(asset_root)
    audit = InMemoryAuditLog()
    media = FfmpegMediaTool()

    # LLM 客户端
    if settings.llm.api_key is None:
        raise RuntimeError(
            "开发档需要配置 LLM API 密钥。请在 .env.dev 中设置 VF_LLM__API_KEY"
        )
    llm = AnthropicClient(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key.get_secret_value(),
        timeout=settings.llm.timeout_seconds,
    )

    # 提示词注册表并激活
    prompts = InMemoryPromptRegistry(audit=audit)
    seed_and_activate_prompts(prompts, actor="bootstrap")

    return DevContainer(
        settings=settings,
        capabilities=build_capabilities(
            CapabilityDeps(
                artifacts=artifacts,
                assets=assets,
                probe=media,
                audio=media,
                shots=SceneDetectShotDetector(),
                frames=media,
                # 模型懒加载：首次转写时才下载，不拖慢进程启动。
                recognizer=FasterWhisperRecognizer(),
                llm=llm,
                prompts=prompts,
                vision_model=settings.llm.vision_model,
            )
        ),
        artifacts=artifacts,
        assets=assets,
        lineage=lineage,
        idempotency=InMemoryIdempotencyStore(),
        audit=audit,
        prompts=prompts,
    )


def create_dev_app(asset_root: Path | None = None) -> FastAPI:
    """uvicorn 入口。"""
    from app.api.app import create_app

    settings = load_settings()
    root = asset_root if asset_root is not None else Path(".local_assets")
    root.mkdir(parents=True, exist_ok=True)
    return create_app(build_dev_container(settings, root), enable_dev_ui=True)
