#!/usr/bin/env bash
#
# 唯一的校验入口。本地和 CI 跑的是同一个脚本。
#
# 这一点很重要：如果 CI 在 workflow 里另写一遍命令列表，两边迟早会漂移——
# 通常表现为「本地过了 CI 不过」，或更糟的「CI 过了但本地那条检查早就没在跑」。
#
# 用法：
#   bash scripts/check.sh          # 静态检查 + 单元/契约测试
#   bash scripts/check.sh --all    # 追加集成测试（需要 docker compose up -d）

set -euo pipefail
cd "$(dirname "$0")/.."

run_integration=false
if [ "${1:-}" = "--all" ]; then
    run_integration=true
fi

step() {
    printf '\n\033[1m▸ %s\033[0m\n' "$1"
}

step "ruff check"
ruff check .

step "ruff format --check"
ruff format --check .

step "mypy --strict"
mypy

step "import-linter 边界契约"
lint-imports

step "边界守卫自检（注入违规探针）"
bash scripts/verify_boundaries.sh

step "单元与契约测试"
pytest

if [ "$run_integration" = true ]; then
    step "集成测试"
    pytest -m integration
fi

printf '\n\033[32m全部检查通过\033[0m\n'
