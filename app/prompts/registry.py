"""提示词注册表的内存实现 + Jinja2 渲染。

渲染用 `StrictUndefined`：模板里出现未提供的变量时**报错**，而不是渲染成空字符串。
后者的后果是模型收到一个缺了关键上下文的提示词，仍然会给出一份看起来合理的输出，
而这种错不会有任何症状。
"""

from __future__ import annotations

from datetime import UTC, datetime

from jinja2 import Environment, StrictUndefined, TemplateError, meta

from app.domain.idempotency import AuditAction, AuditLog
from app.domain.prompt_registry import (
    PromptRenderError,
    PromptRequirementReport,
    PromptStatus,
    PromptTemplate,
    PromptVersion,
)
from app.domain.prompts import ResolvedPrompt


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryPromptRegistry:
    def __init__(self, audit: AuditLog | None = None) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._versions: dict[str, list[PromptVersion]] = {}
        self._audit = audit
        self._env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701

    # ------------------------------------------------------------------ 登记

    def register(self, template: PromptTemplate) -> PromptTemplate:
        existing = self._templates.get(template.key)
        if existing is not None and existing != template:
            # 契约变了意味着调用方的注入代码也要改，静默覆盖会让两边错位。
            raise PromptRenderError(
                f"键位 {template.key} 已登记且契约不同，"
                f"变量声明由 {existing.variables} 变为 {template.variables}"
            )
        self._templates.setdefault(template.key, template)
        self._versions.setdefault(template.key, [])
        return self._templates[template.key]

    def templates(self) -> tuple[PromptTemplate, ...]:
        return tuple(self._templates.values())

    def templates_by_stage(self) -> dict[str, tuple[PromptTemplate, ...]]:
        grouped: dict[str, list[PromptTemplate]] = {}
        for template in self._templates.values():
            grouped.setdefault(template.stage, []).append(template)
        return {stage: tuple(items) for stage, items in sorted(grouped.items())}

    # ------------------------------------------------------------------ 版本

    def add_version(self, key: str, body: str, *, change_note: str, author: str) -> PromptVersion:
        template = self._require_template(key)
        self._validate_body(template, body)

        chain = self._versions[key]
        version = PromptVersion(
            key=template.key,
            version=len(chain) + 1,
            body=body,
            # 一律草案落地。开发把草案写进注册表这个动作因此是安全的：草案跑不起来。
            status=PromptStatus.DRAFT,
            change_note=change_note,
            author=author,
            created_at=_now(),
        )
        chain.append(version)
        return version

    def activate(self, key: str, version: int, *, actor: str) -> PromptVersion:
        target = self.get_version(key, version)
        if target.status is PromptStatus.ACTIVE:
            return target

        previous = self.active_version(key)
        if previous is not None:
            previous.status = PromptStatus.RETIRED

        target.status = PromptStatus.ACTIVE
        target.activated_at = _now()
        target.activated_by = actor

        if self._audit is not None:
            self._audit.record(
                AuditAction.PROMPT_CHANGED,
                actor=actor,
                subject=f"prompt:{key}",
                summary=(
                    f"激活 v{version}"
                    + (f"（取代 v{previous.version}）" if previous else "")
                    + f"：{target.change_note}"
                ),
            )
        return target

    def history(self, key: str) -> tuple[PromptVersion, ...]:
        self._require_template(key)
        return tuple(self._versions[key])

    def get_version(self, key: str, version: int) -> PromptVersion:
        chain = self._versions.get(key, [])
        if not 1 <= version <= len(chain):
            raise PromptRenderError(f"提示词版本不存在：{key}@v{version}")
        return chain[version - 1]

    def active_version(self, key: str) -> PromptVersion | None:
        for version in self._versions.get(key, []):
            if version.status is PromptStatus.ACTIVE:
                return version
        return None

    # ------------------------------------------------------------------ 渲染

    def resolve(self, key: str, variables: dict[str, object]) -> ResolvedPrompt:
        active = self.active_version(key)
        if active is None:
            drafts = [v.version for v in self._versions.get(key, [])]
            hint = f"，当前有草案 {drafts}，需业务评审后激活" if drafts else ""
            raise PromptRenderError(f"提示词 {key} 没有已激活版本{hint}")
        return self._render(active, variables)

    def resolve_version(
        self, key: str, version: int, variables: dict[str, object]
    ) -> ResolvedPrompt:
        target = self.get_version(key, version)
        if not target.is_usable:
            raise PromptRenderError(f"{key}@v{version} 处于 {target.status} 状态，不可用于生产")
        return self._render(target, variables)

    def preview(self, key: str, version: int, variables: dict[str, object]) -> ResolvedPrompt:
        """预览任意版本，**包括草案**。

        配置管理台的试运行面板用它。与 `resolve_*` 分开命名是刻意的：预览是给人看的，
        解析是给生产用的，两者混成一个函数迟早会有人用预览的结果去跑生产。
        """
        return self._render(self.get_version(key, version), variables)

    def requirement_report(self, key: str) -> PromptRequirementReport:
        template = self._templates.get(key)
        if template is None:
            return PromptRequirementReport(
                key=key, registered=False, active_version=None, draft_versions=()
            )
        active = self.active_version(key)
        return PromptRequirementReport(
            key=template.key,
            registered=True,
            active_version=active.version if active else None,
            draft_versions=tuple(
                v.version for v in self._versions[key] if v.status is PromptStatus.DRAFT
            ),
        )

    # ------------------------------------------------------------------ 内部

    def _require_template(self, key: str) -> PromptTemplate:
        template = self._templates.get(key)
        if template is None:
            raise PromptRenderError(f"提示词键位未登记：{key}")
        return template

    def _validate_body(self, template: PromptTemplate, body: str) -> None:
        """正文里用到的变量必须都在契约里声明过。

        在写入时就校验，而不是等渲染时才发现：草案入库和激活之间可能隔很久，
        那时作者已经不在上下文里了。
        """
        try:
            used = meta.find_undeclared_variables(self._env.parse(body))
        except TemplateError as exc:
            raise PromptRenderError(f"{template.key} 模板语法错误：{exc}") from exc

        undeclared = sorted(used - set(template.variables))
        if undeclared:
            raise PromptRenderError(
                f"{template.key} 正文使用了未声明的变量 {undeclared}，"
                f"已声明的是 {list(template.variables)}"
            )

    def _render(self, version: PromptVersion, variables: dict[str, object]) -> ResolvedPrompt:
        template = self._templates[version.key]
        missing = sorted(set(template.variables) - set(variables))
        if missing:
            raise PromptRenderError(f"{version.key} 渲染缺少变量 {missing}")

        try:
            text = self._env.from_string(version.body).render(**variables)
        except TemplateError as exc:
            raise PromptRenderError(f"{version.key} 渲染失败：{exc}") from exc

        if not text.strip():
            raise PromptRenderError(f"{version.key} 渲染结果为空")

        return ResolvedPrompt(key=version.key, version=version.version, text=text)
