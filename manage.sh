#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
  cat <<'USAGE'
Usage: ./manage.sh <command> [options]

Commands:
  serve   Start the FastAPI app with uvicorn (options: --host, --port, --reload, ...)
  train   Run an ASL training module (use --dataset letters|words, defaults to letters)
  clean   Remove training artifacts (runs directory and generated weights)

Examples:
  ./manage.sh serve --reload
  ./manage.sh train --dataset letters|words --epochs 50 --batch-size 16 --device mps|cpu|cuda
  ./manage.sh clean
USAGE
}

cmd_serve() {
  local host="127.0.0.1"
  local port="8000"
  local reload="false"
  local forwarded=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host)
        [[ $# -lt 2 ]] && { echo "Missing value for --host" >&2; exit 1; }
        host="$2"
        shift 2
        ;;
      --port)
        [[ $# -lt 2 ]] && { echo "Missing value for --port" >&2; exit 1; }
        port="$2"
        shift 2
        ;;
      --reload)
        reload="true"
        shift
        ;;
      *)
        forwarded+=("$1")
        shift
        ;;
    esac
  done

  local cmd=("python" "-m" "uvicorn" "app:app" "--host" "$host" "--port" "$port")
  [[ "$reload" == "true" ]] && cmd+=("--reload")
  if (( ${#forwarded[@]} )); then
    cmd+=("${forwarded[@]}")
  fi

  exec "${cmd[@]}"
}

cmd_train() {
  local dataset="letters"
  local forwarded=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset)
        [[ $# -lt 2 ]] && { echo "Missing value for --dataset" >&2; exit 1; }
        dataset="$2"
        shift 2
        ;;
      *)
        forwarded+=("$1")
        shift
        ;;
    esac
  done

  local module=""
  case "$dataset" in
    letters)
      module="sign_language.training.asl_letters_v2.trainer"
      ;;
    words)
      module="sign_language.training.asl_words.trainer"
      ;;
    *)
      echo "Unknown dataset '$dataset'. Use 'letters' or 'words'." >&2
      exit 1
      ;;
  esac

  local cmd=("python" "-m" "$module")
  if (( ${#forwarded[@]} )); then
    cmd+=("${forwarded[@]}")
  fi

  exec "${cmd[@]}"
}

cmd_clean() {
  local runs_dir="${SCRIPT_DIR}/sign_language/training/runs"
  local weights_path="${SCRIPT_DIR}/models/asl-sign-detector.pt"
  local weights_words_path="${SCRIPT_DIR}/models/asl-words-detector.pt"
  local cache_dirs=(
    "${SCRIPT_DIR}/sign_language/training/asl_letters_v2"
    "${SCRIPT_DIR}/sign_language/training/asl_words"
  )

  echo "This will remove training artifacts:"
  echo "  - ${runs_dir}"
  echo "  - ${weights_path}"
  echo "  - ${weights_words_path}"
  for dir in "${cache_dirs[@]}"; do
    echo "  - ${dir}/*/labels.cache"
  done
  read -r -p "Proceed? [y/N]: " response

  case "$response" in
    [yY][eE][sS]|[yY])
      if [[ -d "$runs_dir" ]]; then
        rm -rf "$runs_dir"
        echo "Removed $runs_dir"
      else
        echo "No runs directory found at $runs_dir"
      fi

      if [[ -f "$weights_path" ]]; then
        rm -f "$weights_path"
        echo "Removed $weights_path"
      else
        echo "No weights file found at $weights_path"
      fi

      if [[ -f "$weights_words_path" ]]; then
        rm -f "$weights_words_path"
        echo "Removed $weights_words_path"
      else
        echo "No weights file found at $weights_words_path"
      fi

      local removed=false
      shopt -s nullglob
      for dir in "${cache_dirs[@]}"; do
        for cache_path in "${dir}"/*/labels.cache; do
          rm -f "$cache_path"
          echo "Removed $cache_path"
          removed=true
        done
      done
      shopt -u nullglob
      if [[ "$removed" = false ]]; then
        echo "No label cache files found"
      fi
      ;;
    *)
      echo "Aborted."
      ;;
  esac
}

[[ $# -lt 1 ]] && { print_usage; exit 1; }

case "$1" in
  serve)
    shift
    cmd_serve "$@"
    ;;
  train)
    shift
    cmd_train "$@"
    ;;
  clean)
    shift
    cmd_clean "$@"
    ;;
  --help|-h|help)
    print_usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    print_usage >&2
    exit 1
    ;;
esac
