#!/usr/bin/env bash
# 每节点执行一次；参数可在这里编辑，也可用环境变量覆盖。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
REPO_DIR="$PWD"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CORPUS="${CORPUS:-/volume/data/hjiang02/open_source/datasets/fineweb/sample/100BT}"
TOKENS="${TOKENS:-}"
RATIO="${RATIO:-}"
if [[ -n "$TOKENS" && -n "$RATIO" ]]; then
    echo 'TOKENS 与 RATIO 只能设置一个' >&2; exit 2
fi
if [[ -n "$RATIO" ]]; then budget_args=(--ratio "$RATIO"); else budget_args=(--tokens "${TOKENS:-1e8}"); fi

args=(
    --model "${MODEL:-models/configs/pythia-14m}" --name "${RUN_NAME:-14m-d100m}"
    --seq "${SEQ:-2048}" --batch "${BATCH:-8}" --grad-accum "${GRAD_ACCUM:-1}"
    --lr "${LR:-6e-4}" --num-workers "${NUM_WORKERS:-8}"
    --world-size-unit "${WORLD_SIZE_UNIT:-nodes}" --run-root "${RUN_ROOT:-$REPO_DIR/runs}"
    --swanlab-mode "${SWANLAB_MODE:-local}" --swanlab-project "${SWANLAB_PROJECT:-scalinglaw}"
)
if [[ -n "${WARMUP_STEPS:-}" ]]; then args+=(--warmup-steps "$WARMUP_STEPS"); fi
if [[ -n "${SAVE_STEPS:-}" ]]; then args+=(--save-steps "$SAVE_STEPS"); fi
if [[ "${GRAD_CHECKPOINTING:-0}" == 1 ]]; then args+=(--grad-checkpointing); fi
if [[ "${SAVE_FINAL:-0}" == 1 ]]; then args+=(--save-final); fi
if [[ "${KEEP_MODEL:-0}" == 1 ]]; then args+=(--keep-model); fi
if [[ "${ALLOW_REUSE:-0}" == 1 ]]; then args+=(--allow-reuse); fi

extra_args=()
plot_args=()
dry_run=0
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --tokens|--ratio|--corpus|--title)
            if (( $# < 2 )); then echo "$1 缺少参数值" >&2; exit 2; fi
            case "$1" in
                --tokens|--ratio) budget_args=("$1" "$2") ;;
                --corpus) CORPUS="$2" ;;
                --title) plot_args+=(--title "$2") ;;
            esac
            shift 2 ;;
        --tokens=*|--ratio=*) budget_args=("$1"); shift ;;
        --corpus=*) CORPUS="${1#*=}"; shift ;;
        --title=*) plot_args+=(--title "${1#*=}"); shift ;;
        --help|-h) exec "$PYTHON_BIN" scripts/train.py --help ;;
        *) extra_args+=("$1"); shift ;;
    esac
done
args+=("${budget_args[@]}" "${extra_args[@]}")

# 预算预览不训练，用于向数据准备步骤传递精确 token 数。
budget_json="$("$PYTHON_BIN" scripts/train.py "${args[@]}" --dry-run)"
if [[ "$dry_run" == 1 ]]; then
    printf '%s\n' "$budget_json"
    exit 0
fi
plan_info="$("$PYTHON_BIN" -c 'import json, sys; p=json.load(sys.stdin); print(p["distributed"]["node_rank"]); print(p["run_dir"])' <<< "$budget_json")"
node_rank="${plan_info%%$'\n'*}"
run_dir="${plan_info#*$'\n'}"

if [[ "$node_rank" == 0 ]]; then
    echo '[1/3] 准备数据：corpus.py' >&2
    "$PYTHON_BIN" scripts/corpus.py --corpus "$CORPUS" --budget-file - <<< "$budget_json"
fi
echo '[2/3] 训练：train.py' >&2
"$PYTHON_BIN" scripts/train.py "${args[@]}"
if [[ "$node_rank" == 0 ]]; then
    echo '[3/3] 出图：plot_run.py' >&2
    "$PYTHON_BIN" scripts/plot_run.py "$run_dir" "${plot_args[@]}"
fi
