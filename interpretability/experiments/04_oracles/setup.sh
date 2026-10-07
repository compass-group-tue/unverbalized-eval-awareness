#!/usr/bin/env bash
# Build the Activation Oracle checkout: upstream at the pinned commit, our patch,
# then our overlay files. Sets up $AO_ROOT for the cluster/ wrappers.
#
#   bash experiments/04_oracles/setup.sh   # from interpretability/
#   export AO_ROOT=$PWD/experiments/04_oracles/activation_oracles
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/activation_oracles"
PIN=9816813552346b519f7e5c924a3e3204d5b723c6

[ -d "$DEST/.git" ] || git clone https://github.com/adamkarvonen/activation_oracles "$DEST"
git -C "$DEST" checkout -q "$PIN"
git -C "$DEST" apply --check "$HERE/upstream.patch" 2>/dev/null && git -C "$DEST" apply "$HERE/upstream.patch"
cp -R "$HERE/overlay/." "$DEST/"
mkdir -p "$DEST/cluster/logs"
echo "ready: export AO_ROOT=$DEST"
