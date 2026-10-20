#!/usr/bin/env bash
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo"
task_python="$task_repo/.venv/bin/python"
"$task_python" scripts/kernel_overnight.py --preflight "$@"
mkdir -p analysis/kernel/overnight
nohup "$task_python" -u scripts/kernel_overnight.py "$@" \
  > analysis/kernel/overnight/launch.log 2>&1 < /dev/null &
task_pid=$!
printf 'Overnight runner started: PID %s\n' "$task_pid"
printf 'Progress: analysis/kernel/overnight/launch.log\n'
printf 'Morning report: analysis/kernel/overnight/SUMMARY.md\n'
printf 'The pod stays running. Stop it yourself after checking the persistent backup.\n'
