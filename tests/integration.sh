#!/usr/bin/env bash
# Integration smoke tests for the gizmo binary.
# These do not require a model; they verify CLI plumbing and error paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIZMO_BIN="${GIZMO_BIN:-${SCRIPT_DIR}/../gizmo}"

if [[ ! -x "${GIZMO_BIN}" ]]; then
    echo "[SKIP] gizmo binary not found at ${GIZMO_BIN} (set GIZMO_BIN)"
    exit 0
fi

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

"${GIZMO_BIN}" --help >"${TMPDIR}/help.txt" 2>&1

echo "[TEST] help mentions context-size"
grep -q -- "--context-size" "${TMPDIR}/help.txt"
grep -q -- "GIZMO_CONTEXT_SIZE" "${TMPDIR}/help.txt"

echo "[TEST] run with missing model fails"
if "${GIZMO_BIN}" run -m "${TMPDIR}/missing.gguf" -p "hello" -c 128 2>"${TMPDIR}/run.err"; then
    echo "[FAIL] expected missing-model run to fail"
    exit 1
fi
grep -qi "failed to load\|error" "${TMPDIR}/run.err"

echo "[TEST] launch claude help works"
"${GIZMO_BIN}" launch claude --help >"${TMPDIR}/launch_help.txt" 2>&1
grep -qi "claude\|launch" "${TMPDIR}/launch_help.txt"

echo "[TEST] context-size flag is parsed (invalid model still fails before inference)"
if "${GIZMO_BIN}" run -m "${TMPDIR}/missing.gguf" -p "hello" -c 65536 2>"${TMPDIR}/ctx.err"; then
    echo "[FAIL] expected missing-model run to fail"
    exit 1
fi
# The context size is parsed and used; the failure must be about the model,
# not about an unknown flag.
if grep -q -- "unknown\|unrecognized\|invalid" "${TMPDIR}/ctx.err"; then
    echo "[FAIL] context-size flag rejected as unknown"
    exit 1
fi

echo "[PASS] all integration smoke tests passed"
