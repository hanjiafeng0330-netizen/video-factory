#!/usr/bin/env bash
#
# 边界守卫自检 —— 验证 T0-1 的验收标准。
#
# import-linter 报告「契约全部 KEPT」并不足以证明守卫有效：一个配置错误的
# 契约同样会显示 KEPT。本脚本为每条契约注入一个真实违规的探针模块，断言
# lint-imports 必须失败且对应契约必须 BROKEN，随后移除探针。
#
# 任一探针未能触发失败，说明该条契约形同虚设，脚本以非零码退出。
#
# 用法：bash scripts/verify_boundaries.sh

set -uo pipefail
cd "$(dirname "$0")/.."

if ! command -v lint-imports >/dev/null 2>&1; then
    echo "ERROR: lint-imports 不在 PATH 中，请先激活虚拟环境" >&2
    exit 2
fi

PROBE=""
cleanup() {
    [ -n "$PROBE" ] && rm -f "$PROBE"
    find app -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
    return 0
}
trap cleanup EXIT

failures=0

# 探针定义：<探针文件路径>|<违规 import 语句>|<预期被打破的契约名>
probes=(
    "app/capabilities/preprocess/_boundary_probe.py|import app.capabilities.understand|能力模块互不依赖"
    "app/capabilities/script_gen/_boundary_probe.py|import app.workflow|能力模块只能依赖 domain/adapters/assets"
    "app/adapters/llm/_boundary_probe.py|import app.platform|适配层只能依赖 domain"
    "app/api/_boundary_probe.py|import app.adapters.llm|API 层不得直连适配层"
    "app/domain/script/_boundary_probe.py|import sqlalchemy|domain 层无框架依赖"
)

for probe in "${probes[@]}"; do
    IFS='|' read -r path statement contract <<< "$probe"
    PROBE="$path"

    printf '%s\n' "$statement" > "$path"
    output=$(lint-imports 2>&1)
    status=$?

    rm -f "$path"
    PROBE=""

    if [ "$status" -eq 0 ]; then
        echo "FAIL  [$contract] 注入违规后 lint-imports 仍然通过——该契约未生效"
        failures=$((failures + 1))
        continue
    fi

    if ! printf '%s' "$output" | grep -qF "$contract BROKEN"; then
        echo "FAIL  [$contract] lint-imports 失败了，但打破的不是预期契约"
        printf '%s\n' "$output" | grep -E 'KEPT|BROKEN' | sed 's/^/        /'
        failures=$((failures + 1))
        continue
    fi

    echo "OK    [$contract] 违规被拦截"
done

echo
if [ "$failures" -ne 0 ]; then
    echo "边界守卫自检失败：$failures 条契约未能拦截违规"
    exit 1
fi

# 探针全部移除后，基线必须是干净的
if ! lint-imports >/dev/null 2>&1; then
    echo "边界守卫自检失败：移除探针后基线仍不干净"
    exit 1
fi

echo "边界守卫自检通过：5 条契约均可拦截违规，且基线干净"
