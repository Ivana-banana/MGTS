#!/usr/bin/env bash
# Запись области сцены RViz в контейнере организаторов на Linux.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 || ! "$1" =~ ^(top|third-person)$ || "$3" != *.mp4 || ( $# -eq 4 && ! "$4" =~ ^[0-9]+$ ) ]]; then
    echo "Использование: $0 {top|third-person} КАТАЛОГ_ПРОГОНА ВЫХОД.mp4 [СЕКУНДА_ПРЕДПРОСМОТРА]" >&2
    exit 2
fi
view=$1
input=$(realpath -- "$2")
output=$(realpath -m -- "$3")
preview_offset=${4:-}
project=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
image=${MGTS_ROS_IMAGE:-hakatonius:latest}
wall_top=1.35
wall_alpha=1.0
if [[ "$view" = third-person ]]; then
    wall_top=0.60
    wall_alpha=0.45
fi
test -f "$input/map.npz"
test -f "$input/bag/metadata.yaml"
test ! -e "$output"
mkdir -p -- "$(dirname -- "$output")"
scratch=$(mktemp -d --tmpdir="$(dirname -- "$output")" .capture-go2.XXXXXX)
mkdir -m 700 -- "$scratch/home" "$scratch/runtime"
finish() {
    if [[ -s "$scratch/container" ]]; then
        docker stop --time 2 "$(cat "$scratch/container")" >/dev/null 2>&1 || true
    fi
    if [[ -n ${recorder_pid:-} ]]; then
        kill -INT "$recorder_pid" 2>/dev/null || true
    fi
    if [[ -n ${xvfb_pid:-} ]]; then
        kill "$xvfb_pid" 2>/dev/null || true
    fi
    rm -rf -- "$scratch"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

Xvfb -displayfd 3 -screen 0 1280x800x24 -nolisten tcp 3>"$scratch/display" \
    >"$scratch/xvfb.log" 2>&1 &
xvfb_pid=$!
for ((attempt = 0; attempt < 100; attempt++)); do
    [[ -s "$scratch/display" ]] && break
    kill -0 "$xvfb_pid" || { cat "$scratch/xvfb.log" >&2; exit 1; }
    sleep 0.1
done
test -s "$scratch/display"
export DISPLAY=":$(cat "$scratch/display")"

docker run --pull never --rm --cidfile "$scratch/container" --user "$(id -u):$(id -g)" \
    -e DISPLAY -e HOME=/capture/home -e XDG_RUNTIME_DIR=/capture/runtime \
    -e ROS_LOCALHOST_ONLY=1 -e LIBGL_ALWAYS_SOFTWARE=1 -e LP_NUM_THREADS=2 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$project:/mgts:ro" \
    -v "$input:/input:ro" -v "$scratch:/capture" "$image" \
    bash -euc '
        python3 -u /mgts/tools/rviz_go2.py /input/map.npz --wall-top "$3" --wall-alpha "$4" &
        visual_pid=$!
        rviz2 -d "/mgts/configs/go2-surface-$1.rviz" &
        rviz_pid=$!
        while [[ ! -f /capture/start ]]; do
            kill -0 "$visual_pid" "$rviz_pid"
            sleep 0.1
        done
        player_options=()
        if [[ -n "$2" ]]; then
            player_options+=(--start-offset "$2")
        fi
        ros2 bag play /input/bag --rate 1 --disable-keyboard-controls \
            --qos-profile-overrides-path /mgts/configs/go2-result-qos.yaml "${player_options[@]}"
        kill -0 "$visual_pid" "$rviz_pid"
        sleep 1
        kill -TERM "$visual_pid"
        wait "$visual_pid"
        touch /capture/end
        wait "$rviz_pid"
    ' bash "$view" "$preview_offset" "$wall_top" "$wall_alpha" >"${output%.mp4}.ros.log" 2>&1 &
container_pid=$!

viewport() {
    # Qt names RViz's native renderer "rviz2". Match its mapped, full-size window.
    local id geometry left top width height
    for id in $(xwininfo -root -tree | awk '/"rviz2": .* [0-9]+x[0-9]+[+-]/ { print $1 }'); do
        geometry=$(xwininfo -id "$id")
        read -r left top width height < <(awk '
            /Absolute upper-left X:/ { x=$NF }
            /Absolute upper-left Y:/ { y=$NF }
            /Width:/ { w=$NF }
            /Height:/ { h=$NF }
            END { print x,y,w,h }
        ' <<<"$geometry")
        if ((width >= 640 && height >= 360)); then
            printf '%s %s %s %s\n' "$left" "$top" "$width" "$height"
            return
        fi
    done
    return 1
}
for ((attempt = 0; attempt < 150; attempt++)); do
    if read -r left top width height < <(viewport); then
        break
    fi
    kill -0 "$container_pid" || { tail -30 "${output%.mp4}.ros.log" >&2; exit 1; }
    sleep 0.1
done
[[ ${width:-0} -ge 640 ]] || { echo "Область сцены RViz не открылась." >&2; exit 1; }

# Fold the Displays sidebar for capture, and switch to RViz's own fullscreen mode.
if ((left > 100)); then
    xdotool mousemove "$((left - 18))" "$((top + height / 2))" click 1
fi
window=$(xdotool search --onlyvisible --name ' - RViz$' | head -n 1)
xdotool windowfocus "$window" key F11 mousemove 0 0
sleep 0.5
read -r left top width height < <(viewport)
if ((width < 1200 || left < 0 || top < 0 || left + width > 1280 || top + height > 800)); then
    echo "Размер сцены RViz вне области захвата: ${width}x${height}+$left+$top" >&2
    exit 1
fi
width=$((width - width % 2))
height=$((height - height % 2))
capture_options=()
if [[ -n "$preview_offset" ]]; then
    capture_options+=(-t 7)
fi

ffmpeg -n -nostdin -loglevel warning -f x11grab -framerate 24 -draw_mouse 0 \
    -video_size "${width}x${height}" -i "$DISPLAY.0+$left,$top" \
    -an -c:v libx264 -threads 2 -preset veryfast -crf 19 -pix_fmt yuv420p \
    -movflags +faststart "${capture_options[@]}" "$scratch/scene.mp4" >"${output%.mp4}.ffmpeg.log" 2>&1 &
recorder_pid=$!
sleep 0.35
touch "$scratch/start"
while [[ ! -f "$scratch/end" ]]; do
    kill -0 "$container_pid" || { tail -30 "${output%.mp4}.ros.log" >&2; exit 1; }
    if ! kill -0 "$recorder_pid" 2>/dev/null; then
        wait "$recorder_pid"
        [[ -n "$preview_offset" ]] || { echo "Запись закончилась раньше воспроизведения." >&2; exit 1; }
        break
    fi
    sleep 0.2
done
recorder_status=0
if kill -0 "$recorder_pid" 2>/dev/null; then
    kill -INT "$recorder_pid"
fi
wait "$recorder_pid" || recorder_status=$?
recorder_pid=
[[ $recorder_status -eq 0 || $recorder_status -eq 255 ]]
test -s "$scratch/scene.mp4"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,duration,nb_frames "$scratch/scene.mp4"
ln -T -- "$scratch/scene.mp4" "$output"
