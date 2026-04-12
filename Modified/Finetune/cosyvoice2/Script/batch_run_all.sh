#!/bin/bash
# Batch run CosyVoice2 data preparation and training scripts.
# Each role is a first-level subdirectory under Dataset.
# 00_prepare_data.sh automatically splits a validation set with ratio CV_RATIO.

set -euo pipefail

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONFIRM_BEFORE_RUN=true
CV_RATIO="${CV_RATIO:-0.05}"

PHASE_SCRIPTS=(
  "00_prepare_data.sh"
  "01_extract_embedding.sh"
  "02_extract_speech_token.sh"
  "03_make_parquet.sh"
  "04_train.sh"
)
PHASE_DESCRIPTIONS=(
  "Data preparation"
  "Extract speaker embeddings"
  "Extract speech tokens"
  "Make parquet"
  "Train"
)

check_scripts() {
  echo -e "${BLUE}[INFO]${NC} Checking phase scripts..."
  for script_name in "${PHASE_SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script_name"
    if [[ ! -f "$script_path" ]]; then
      echo -e "${RED}[ERROR]${NC} Missing script: $script_path" >&2
      exit 1
    fi
    if [[ ! -x "$script_path" ]]; then
      chmod +x "$script_path" || true
    fi
  done
  echo -e "${GREEN}[OK]${NC} Script check passed"
}

get_roles() {
  local ds_dir="$ROOT_DIR/Dataset"
  local roles=()

  if [[ ! -d "$ds_dir" ]]; then
    echo -e "${RED}[ERROR]${NC} Dataset directory not found: $ds_dir" >&2
    exit 1
  fi

  shopt -s nullglob
  for dir in "$ds_dir"/*; do
    [[ -d "$dir" ]] || continue
    role_name=$(basename "$dir")
    [[ "$role_name" == .* ]] && continue
    roles+=("$role_name")
  done
  shopt -u nullglob

  if [[ ${#roles[@]} -eq 0 ]]; then
    echo -e "${RED}[ERROR]${NC} No roles found under Dataset" >&2
    exit 1
  fi

  printf "%s\n" "${roles[@]}"
}

show_summary() {
  local roles=("$@")
  echo -e "\n${BLUE}========================================${NC}"
  echo -e "${BLUE}Batch Execution Summary${NC}"
  echo -e "${BLUE}========================================${NC}"
  echo -e "${BLUE}Role count:${NC} ${#roles[@]}"
  echo -e "${BLUE}Validation ratio:${NC} ${CV_RATIO}"
  echo -e "${BLUE}Stages:${NC}"
  for idx in "${!PHASE_DESCRIPTIONS[@]}"; do
    echo -e "  $((idx + 1)). ${PHASE_DESCRIPTIONS[$idx]} (${PHASE_SCRIPTS[$idx]})"
  done
  echo -e "${BLUE}Roles:${NC}"
  for role_name in "${roles[@]}"; do
    echo -e "  - $role_name"
  done
}

confirm_execution() {
  if [[ "$CONFIRM_BEFORE_RUN" != true ]]; then
    return 0
  fi
  echo -ne "${YELLOW}[CONFIRM]${NC} Continue? (y/N): "
  read -r answer
  case "$answer" in
    [yY]|[yY][eE][sS]) return 0 ;;
    *) echo -e "${YELLOW}[CANCELLED]${NC}"; exit 0 ;;
  esac
}

run_phase_for_role() {
  local role_name="$1"

  for idx in "${!PHASE_SCRIPTS[@]}"; do
    local script_name="${PHASE_SCRIPTS[$idx]}"
    local description="${PHASE_DESCRIPTIONS[$idx]}"
    echo -e "\n${PURPLE}[EXEC]${NC} Role ${CYAN}$role_name${NC} -> $description ($script_name)"
    if [[ "$script_name" == "00_prepare_data.sh" ]]; then
      if "$SCRIPT_DIR/$script_name" "$role_name" "$CV_RATIO"; then
        echo -e "${GREEN}[SUCCESS]${NC} $script_name completed (role: $role_name)"
      else
        echo -e "${RED}[ERROR]${NC} $script_name failed (role: $role_name)"
        return 1
      fi
    else
      if "$SCRIPT_DIR/$script_name" "$role_name"; then
        echo -e "${GREEN}[SUCCESS]${NC} $script_name completed (role: $role_name)"
      else
        echo -e "${RED}[ERROR]${NC} $script_name failed (role: $role_name)"
        return 1
      fi
    fi
    sleep 1
  done
}

main() {
  check_scripts
  mapfile -t roles < <(get_roles)
  show_summary "${roles[@]}"
  confirm_execution

  local succeeded=()
  local failed=()
  local start_time
  start_time=$(date +%s)

  echo -e "\n${GREEN}[START]${NC} Starting batch execution..."
  for role_name in "${roles[@]}"; do
    if run_phase_for_role "$role_name"; then
      succeeded+=("$role_name")
    else
      failed+=("$role_name")
      echo -e "${YELLOW}[INFO]${NC} Role $role_name failed, continuing"
    fi
  done

  local end_time
  end_time=$(date +%s)
  local elapsed=$((end_time - start_time))

  printf "\n${PURPLE}========================================${NC}\n"
  printf "${PURPLE}Batch Execution Finished${NC}\n"
  printf "${PURPLE}========================================${NC}\n"
  printf "${BLUE}Elapsed:${NC} %02d:%02d:%02d\n" $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
  printf "${GREEN}Succeeded:${NC} %d\n" ${#succeeded[@]}
  printf "${RED}Failed:${NC} %d\n" ${#failed[@]}
  if [[ ${#succeeded[@]} -gt 0 ]]; then
    echo -e "${GREEN}Succeeded roles:${NC}"
    for role_name in "${succeeded[@]}"; do
      echo "  - $role_name"
    done
  fi
  if [[ ${#failed[@]} -gt 0 ]]; then
    echo -e "${RED}Failed roles:${NC}"
    for role_name in "${failed[@]}"; do
      echo "  - $role_name"
    done
  fi
}

trap 'echo -e "\n${YELLOW}[INTERRUPTED]${NC} Interrupted"; exit 130' INT

main "$@"
