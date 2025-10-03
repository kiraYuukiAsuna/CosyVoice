#!/bin/bash
# 批量运行 CosyVoice2 数据准备与训练脚本
# 会自动遍历 Dataset 目录下的角色（每个子目录视为一个角色）
# 依次执行：00_prepare_data -> 01_extract_embedding -> 02_extract_speech_token -> 03_make_parquet -> 04_train

set -euo pipefail

# ===== 环境激活（按需启用） =====
# 如需在脚本内激活 conda 环境，请取消注释并将 cosyvoice 替换为你的环境名
# if command -v conda &>/dev/null; then
#   eval "$(conda shell.$(basename "$SHELL") hook)"
#   conda activate cosyvoice
# fi

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

. ./path.sh || exit 1

# 配置
export CUDA_VISIBLE_DEVICES=0
CONFIRM_BEFORE_RUN=true

# 各阶段脚本（按顺序）
PHASE_SCRIPTS=(
  "00_prepare_data.sh"
  "01_extract_embedding.sh"
  "02_extract_speech_token.sh"
  "03_make_parquet.sh"
  "04_train.sh"
)
PHASE_DESCRIPTIONS=(
  "数据准备"
  "提取说话人嵌入"
  "提取语音离散 token"
  "生成 parquet"
  "训练"
)

# 检查 Script 中脚本存在性与可执行权限
check_scripts() {
  echo -e "${BLUE}[INFO]${NC} 检查阶段脚本..."
  for s in "${PHASE_SCRIPTS[@]}"; do
    local p="$SCRIPT_DIR/$s"
    if [[ ! -f "$p" ]]; then
      echo -e "${RED}[ERROR]${NC} 缺少脚本: $p" >&2
      exit 1
    fi
    if [[ ! -x "$p" ]]; then
      chmod +x "$p" || true
    fi
  done
  echo -e "${GREEN}[OK]${NC} 脚本检查通过"
}

# 获取所有角色名
get_roles() {
  local roles=()
  local ds_dir="$ROOT_DIR/Dataset"
  if [[ ! -d "$ds_dir" ]]; then
    echo -e "${RED}[ERROR]${NC} Dataset 目录不存在: $ds_dir" >&2
    exit 1
  fi

  # 将 Dataset 下所有一级子目录视为角色
  shopt -s nullglob
  for d in "$ds_dir"/*; do
    [[ -d "$d" ]] || continue
    local base=$(basename "$d")
    # 忽略隐藏目录
    [[ "$base" == .* ]] && continue
    roles+=("$base")
  done
  shopt -u nullglob

  if [[ ${#roles[@]} -eq 0 ]]; then
    echo -e "${RED}[ERROR]${NC} 未发现任何角色（请在 Dataset 下创建角色子目录）" >&2
    exit 1
  fi
  # 一行一个角色，便于 mapfile -t 正确解析为多个元素
  printf "%s\n" "${roles[@]}"
}

show_summary() {
  local roles=("$@")
  echo -e "\n${BLUE}========================================${NC}"
  echo -e "${BLUE}批量执行摘要${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}角色数量:${NC} ${#roles[@]}"
  echo -e "${BLUE}阶段:${NC}"
  for i in "${!PHASE_DESCRIPTIONS[@]}"; do
    echo -e "  $((i+1)). ${PHASE_DESCRIPTIONS[$i]} (${PHASE_SCRIPTS[$i]})"
  done
  echo -e "${BLUE}角色列表:${NC}"
  for r in "${roles[@]}"; do
    echo -e "  - $r"
  done
}

confirm_execution() {
  if [[ "$CONFIRM_BEFORE_RUN" != true ]]; then
    return 0
  fi
  echo -ne "${YELLOW}[CONFIRM]${NC} 是否继续执行? (y/N): "
  read -r ans
  case "$ans" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) echo -e "${YELLOW}[CANCELLED]${NC}"; exit 0 ;;
  esac
}

run_phase_for_role() {
  local role="$1"
  for i in "${!PHASE_SCRIPTS[@]}"; do
    local s="${PHASE_SCRIPTS[$i]}"
    local d="${PHASE_DESCRIPTIONS[$i]}"
    echo -e "\n${PURPLE}[EXEC]${NC} 角色 ${CYAN}$role${NC} -> $d ($s)"
    if "$SCRIPT_DIR/$s" "$role"; then
      echo -e "${GREEN}[SUCCESS]${NC} $s 完成 (角色: $role)"
    else
      echo -e "${RED}[ERROR]${NC} $s 失败 (角色: $role)"
      return 1
    fi
    sleep 1
  done
}

main() {
  check_scripts
  mapfile -t roles < <(get_roles)
  show_summary "${roles[@]}"
  confirm_execution

  local ok=()
  local fail=()
  local start=$(date +%s)

  echo -e "\n${GREEN}[START]${NC} 开始批量执行..."
  for r in "${roles[@]}"; do
    if run_phase_for_role "$r"; then
      ok+=("$r")
    else
      fail+=("$r")
      echo -e "${YELLOW}[INFO]${NC} 角色 $r 失败，继续下一个角色"
    fi
  done

  local end=$(date +%s)
  local dt=$((end-start))
  printf "\n${PURPLE}========================================${NC}\n"
  printf "${PURPLE}批量执行完成${NC}\n"
  printf "${PURPLE}========================================${NC}\n"
  printf "${BLUE}总耗时:${NC} %02d:%02d:%02d\n" $((dt/3600)) $(((dt%3600)/60)) $((dt%60))
  printf "${GREEN}成功角色数:${NC} %d\n" ${#ok[@]}
  printf "${RED}失败角色数:${NC} %d\n" ${#fail[@]}
  if [[ ${#ok[@]} -gt 0 ]]; then
    echo -e "${GREEN}成功角色:${NC}"
    for r in "${ok[@]}"; do echo "  ✓ $r"; done
  fi
  if [[ ${#fail[@]} -gt 0 ]]; then
    echo -e "${RED}失败角色:${NC}"
    for r in "${fail[@]}"; do echo "  ✗ $r"; done
  fi
}

trap 'echo -e "\n${YELLOW}[INTERRUPTED]${NC} 已中断"; exit 130' INT

main "$@"
