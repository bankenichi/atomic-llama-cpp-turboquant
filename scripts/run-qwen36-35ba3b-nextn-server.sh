#!/usr/bin/env bash
# Owner-tuned Qwen3.6 35B-A3B MoE target + embedded NextN. Override MAIN_GGUF
# and MMPROJ for portable deployments. Full-f16 is a diagnostic override, not
# this fork's normal RTX 5080 cache profile.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MAIN_GGUF="${MAIN_GGUF:-${ROOT}/.scratch/qwen3-6-35b-a3b/qwen3-6-35b-a3b-q8_0.gguf}"
export CTK="${CTK:-turbo4}"
export CTV="${CTV:-turbo3}"
export CTKD="${CTKD:-f16}"
export CTVD="${CTVD:-f16}"
export N_CPU_MOE="${N_CPU_MOE:-28}"

exec "${ROOT}/scripts/run-qwen36-27b-nextn-server.sh" "$@"
