#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHON="${PYTHON:-python}"

scripts=()
for script in scripts/*.sh; do
  [[ "$script" == "scripts/smoke_test.sh" ]] && continue
  bash -n "$script"
  scripts+=("$script")
done

printf '%s\0' "${scripts[@]}" \
  | xargs -0 -n1 -P4 bash -c 'bash "$1" --smoke-test' _

echo "all script smoke tests passed"
