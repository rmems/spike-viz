#!/usr/bin/env bash
# Generate only the three rate gain cases; never alter the upstream checkout.
# Requires Rust 1.99 and this repo's Python environment on PATH.
# AXON_ENCODER_DIR=/path/to/axon-encoder scripts/generate_gain_fixtures.sh
set -euo pipefail

AXON_ENCODER_DIR="${AXON_ENCODER_DIR:?set AXON_ENCODER_DIR to an axon-encoder checkout}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="161886c256955d98aaa1b1a4b14cf014910ebe9d"
if [ "$(git -C "$AXON_ENCODER_DIR" rev-parse HEAD)" != "$PIN" ] || \
   [ -n "$(git -C "$AXON_ENCODER_DIR" status --porcelain)" ]; then
  echo "error: axon-encoder must be clean and checked out at $PIN" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git -C "$AXON_ENCODER_DIR" archive "$PIN" | tar -x -C "$WORK"
git -C "$WORK" apply --unidiff-zero "$REPO_ROOT/scripts/axon_gain_export.patch"
export AXON_ENCODER_GIT_SHA="$PIN"
AXON_EXPORTER_PATCH_SHA256="$(sha256sum "$REPO_ROOT/scripts/axon_gain_export.patch")"
export AXON_EXPORTER_PATCH_SHA256="${AXON_EXPORTER_PATCH_SHA256%% *}"

for gain in 0 1 2; do
  stage="$WORK/staging/$gain"
  (cd "$WORK" && cargo run --locked -q --example export_spike_viz -- rate "$stage" "$gain")
  python3 "$REPO_ROOT/scripts/pack_axon_export.py" "$stage" \
    "$REPO_ROOT/fixtures/axon-encoder/rate/shared_sine_gain_${gain}_v1"
done
