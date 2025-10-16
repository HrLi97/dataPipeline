import pandas as pd
import json
import os

# === 配置路径 ===
csv_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/data/open_human_all/all_csv_local/OpenHumanVid_part_001.sample10.csv"
json_path = "/mnt/cfs/shanhai/jyutong/dataprocessor/posedata_process/OpenHumanVid/data_json_nopeople/OpenHumanVid_part_003_video_s4_1.json"
output_csv = "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/openhuman-multiple/data/openhuman_mul/OpenHumanVid_part_001.sample10.csv"  # 输出文件名：原名 + .filtered.csv


# === 1. 读取 JSON 中所有 video_path 到一个 set（用于快速查找）===
with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 假设 json_data 是一个 list of dict
video_paths_to_exclude = {item['video_path'] for item in json_data if 'video_path' in item}

print(f"Loaded {len(video_paths_to_exclude)} video paths from JSON to exclude.")

# === 2. 读取 CSV ===
df = pd.read_csv(csv_path)

# 检查 'path' 列是否存在
if 'path' not in df.columns:
    raise ValueError(f"'path' column not found in {csv_path}")

print(f"Original CSV has {len(df)} rows.")

# === 3. 过滤：保留 path 不在 video_paths_to_exclude 中的行 ===
filtered_df = df[~df['path'].isin(video_paths_to_exclude)]

print(f"Filtered out {len(df) - len(filtered_df)} rows. Remaining: {len(filtered_df)}")

# === 4. 保存结果 ===
filtered_df.to_csv(output_csv, index=False)
print(f"Filtered CSV saved to: {output_csv}")