#!/usr/bin/env bash
# Pre-render an "object movie" sweep of the OIT scene (yaw x pitch grid) into
# viewer/frames/, plus a manifest.json the HTML viewer reads. Drag-to-orbit then
# just swaps the nearest pre-rendered frame -> smooth orbit with no live GL window
# (which macOS can't provide for GL4.5 OIT anyway).
#
# Usage: ./generate.sh [boxes] [scene] [alpha] [resolve]
#   boxes   : number of cubes            (default 64)
#   scene   : packed|cluster|nested|slabs(default packed)
#   alpha   : per-box opacity            (default 0.4)
#   resolve : fixed|buggy                (default fixed)
set -euo pipefail
cd "$(dirname "$0")"

BOXES=${1:-64}
SCENE=${2:-packed}
ALPHA=${3:-0.4}
RESOLVE=${4:-fixed}

BIN=../build/oit_demo
W=640; H=480
FIXFLAG=""; [ "$RESOLVE" = "fixed" ] && FIXFLAG="--fixed"

# camera distance: scale with the grid so the scene always fits the frame
RADIUS=$(awk -v n="$BOXES" 'BEGIN{ s=int(exp(log(n)/3)+0.999); printf "%.1f", s*2.1*2.0+9 }')

YAWS=(0 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225 240 255 270 285 300 315 330 345)
PITCHES=(-15 5 25 45)

export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless
rm -rf frames && mkdir -p frames

total=$(( ${#YAWS[@]} * ${#PITCHES[@]} ))
n=0
for pi in "${!PITCHES[@]}"; do
  for yi in "${!YAWS[@]}"; do
    out="frames/f_${pi}_${yi}.png"
    "$BIN" "$out" --shaders ../shaders --scene "$SCENE" --boxes "$BOXES" --alpha "$ALPHA" \
        --radius "$RADIUS" --yaw "${YAWS[$yi]}" --pitch "${PITCHES[$pi]}" \
        --size ${W}x${H} $FIXFLAG >/dev/null
    n=$((n+1)); printf "\r  rendered %d/%d" "$n" "$total"
  done
done
echo

# manifest for the HTML viewer
{
  printf '{\n'
  printf '  "boxes": %s, "scene": "%s", "alpha": %s, "resolve": "%s",\n' "$BOXES" "$SCENE" "$ALPHA" "$RESOLVE"
  printf '  "w": %s, "h": %s,\n' "$W" "$H"
  printf '  "yaws": [%s],\n'    "$(IFS=,; echo "${YAWS[*]}")"
  printf '  "pitches": [%s],\n' "$(IFS=,; echo "${PITCHES[*]}")"
  printf '  "pattern": "frames/f_{p}_{y}.png"\n'
  printf '}\n'
} > manifest.json

echo "done -> viewer/manifest.json ($total frames, radius $RADIUS)"
