#!/usr/bin/env bash
# Regenerate the provenance-pinned axon-encoder fixtures.
#
# From a clean checkout:
#   1. Clone Limen-Neural/axon-encoder and check out the pinned commit below.
#   2. Run this script with AXON_ENCODER_DIR pointed at that checkout.
#
#   AXON_ENCODER_DIR=/path/to/axon-encoder scripts/generate_fixtures.sh
#
# The recorded `axon_encoder_git_sha` in every meta.json is the checkout's
# HEAD at generation time. Stochastic cases are reproducible because the
# encoder draws run through seeded `*_with_rng` surfaces; re-running against
# the same SHA produces identical fixtures.
#
# To pin regeneration to a specific axon-encoder commit, set
# AXON_ENCODER_GIT_SHA to the expected SHA — the script then verifies the
# checkout's HEAD matches it before generating.
set -euo pipefail

AXON_ENCODER_DIR="${AXON_ENCODER_DIR:?set AXON_ENCODER_DIR to an axon-encoder checkout}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

if [ "${AXON_ALLOW_DIRTY:-0}" != "1" ] && \
   [ -n "$(git -C "$AXON_ENCODER_DIR" status --porcelain)" ]; then
  echo "error: $AXON_ENCODER_DIR has uncommitted or untracked changes;" >&2
  echo "fixtures must be generated from a clean, committed tree" >&2
  echo "(set AXON_ALLOW_DIRTY=1 to override for local experimentation)." >&2
  exit 1
fi

EXPECTED_SHA="${AXON_ENCODER_GIT_SHA:-}"
HEAD_SHA="$(git -C "$AXON_ENCODER_DIR" rev-parse HEAD)"
if [ -n "$EXPECTED_SHA" ] && [ "$EXPECTED_SHA" != "$HEAD_SHA" ]; then
  echo "error: AXON_ENCODER_GIT_SHA=$EXPECTED_SHA but" >&2
  echo "$AXON_ENCODER_DIR HEAD is $HEAD_SHA" >&2
  exit 1
fi
export AXON_ENCODER_GIT_SHA="$HEAD_SHA"
echo "axon-encoder @ $AXON_ENCODER_GIT_SHA"

ENCODERS="rate poisson latency population temporal predictive"
CASE="shared_sine_v1"

for enc in $ENCODERS; do
  stage="$STAGING/$enc"
  dest="$REPO_ROOT/fixtures/axon-encoder/$enc/$CASE"
  (cd "$AXON_ENCODER_DIR" && cargo run --locked -q --example export_spike_viz -- "$enc" "$stage")
  python3 "$REPO_ROOT/scripts/pack_axon_export.py" "$stage" "$dest"
done

echo "done: fixtures/axon-encoder/<encoder>/$CASE"
