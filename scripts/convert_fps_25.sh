#!/bin/bash

CSV_FILE="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/data/open_human_all/all_csv_local/OpenHumanVid_part_003.sample10.csv"
SRC_ROOT="/mnt/cfs/shanhai/lihaoran/data/OpenHumanVid-final/"
OUT_ROOT="/mnt/cfs/shanhai/lihaoran/data/OpenHumanVid-final/25fps_part/part_003/"
RAY_LOG="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/fps/1027/"

python /mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/openhuman-multiple/pipeline/convert_fps_25.py \
  --csv_file "$CSV_FILE" \
  --src_root "$SRC_ROOT" \
  --out_root "$OUT_ROOT" \
  --ray_log "$RAY_LOG" \
  --fps_required 25.0 \
  --fps_tol 0.05 \
  --num_workers 64 \
  #--is_local 

echo "✅ Done."