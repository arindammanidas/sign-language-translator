#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
  cat <<'USAGE'
Usage: ./manage.sh <command> [options]

Commands:
  serve   Start the FastAPI app with uvicorn (options: --host, --port, --reload, ...)
  train   Run the ASL letters training module (additional args forwarded directly)
  clean   Remove training artifacts (runs directory and generated weights)

Examples:
  ./manage.sh serve --reload
  ./manage.sh train --epochs 50 --batch-size 16
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
  local cmd=("python" "-m" "sign_language.training.asl_letters_v2.trainer")
  if (( $# )); then
    cmd+=("$@")
  fi

  exec "${cmd[@]}"
}

cmd_clean() {
  local runs_dir="${SCRIPT_DIR}/sign_language/training/runs"
  local weights_path="${SCRIPT_DIR}/models/asl-sign-detector.pt"
  local cache_glob="${SCRIPT_DIR}/sign_language/training/asl_letters_v2"/*/labels.cache

  echo "This will remove training artifacts:"
  echo "  - ${runs_dir}"
  echo "  - ${weights_path}"
  echo "  - ${cache_glob}"
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

      local removed=false
      shopt -s nullglob
      for cache_path in $cache_glob; do
        rm -f "$cache_path"
        echo "Removed $cache_path"
        removed=true
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
