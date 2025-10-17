#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/2-out-new/saved_vid_root/visualization"
DST_DIR="/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/2-out-new/saved_vid_root/h264"
mkdir -p "$DST_DIR"

MAX_JOBS=4   # 最大并发 ffmpeg 进程数
FFMPEG_OPTS="-c:v libx264 -preset medium -crf 15 -threads 0 -c:a aac -b:a 128k"

job_count() { jobs -rp | wc -l; }

for src in "$SRC_DIR"/*.{mp4,mov,mkv,avi,flv}; do
  [ -f "$src" ] || continue
  (
    name=$(basename "$src")
    name="${name%.*}"
    dst="$DST_DIR/${name}.mp4"
    echo "转码 $name → h264/${name}.mp4"
    ffmpeg -i "$src" $FFMPEG_OPTS "$dst"
  ) &

  while [ "$(job_count)" -ge "$MAX_JOBS" ]; do
    wait -n
  done
done

wait
echo "全部转码完成，输出目录：$DST_DIR"
