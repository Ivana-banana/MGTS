#!/usr/bin/env bash
# Headless screen recording of RViz with the localization running (inside the mgts-go2 image).
#   record_rviz.sh MAP_DIR BAG VIEW(top|third) OUT.mp4 [RATE] [RECORD_BAG_DIR]
# On Windows with VcXsrv RViz opens on the desktop instead; record it with any screen recorder.
set -euo pipefail
MAP=${1:?map dir}; BAG=${2:?bag dir}; VIEW=${3:-top}; OUT=${4:?output mp4}; RATE=${5:-1.0}; REC=${6:-}
export DISPLAY=:99 LIBGL_ALWAYS_SOFTWARE=1 QT_X11_NO_MITSHM=1
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp >/dev/null 2>&1 &
XPID=$!
sleep 2
ffmpeg -y -loglevel error -f x11grab -video_size 1920x1080 -framerate 15 -i :99 \
       -c:v libx264 -preset veryfast -crf 27 -pix_fmt yuv420p "$OUT" &
FPID=$!
ARGS=(map_dir:="$MAP" bag:="$BAG" view:="$VIEW" rviz:=true rate:="$RATE" start_delay:=6.0)
[ -n "$REC" ] && ARGS+=(record:="$REC")
ros2 launch mgts_go2 localization.launch.py "${ARGS[@]}" || true
kill -INT "$FPID" 2>/dev/null || true
wait "$FPID" 2>/dev/null || true
kill "$XPID" 2>/dev/null || true
echo "video: $OUT"
