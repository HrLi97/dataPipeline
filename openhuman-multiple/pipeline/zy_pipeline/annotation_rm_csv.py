import pandas as pd
import os
import re

def extract_key_from_file_path(path):
    # 分割路径
    parts = path.strip().split('/')
    if len(parts) < 3:
        return None
    dir1, dir2 = parts[-3], parts[-2]
    basename = parts[-1]
    basename_no_ext = os.path.splitext(basename)[0]
    if basename_no_ext.endswith('_vis'):
        basename_no_ext = basename_no_ext[:-4]  # remove '_vis'
    return f"{dir2}/{basename_no_ext}"

def extract_key_from_vid_path(path):
    parts = path.strip().split('/')
    if len(parts) < 3:
        return None
    dir1, dir2 = parts[-3], parts[-2]
    basename = parts[-1]
    basename_no_ext = os.path.splitext(basename)[0]
    return f"{dir2}/{basename_no_ext}"

# 读取 CSV
csv1_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/part-1-moss.csv"
csv2_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/0919_openhuman_one_human_all_25fps_part_1_final.csv"

df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)

# 提取 key 集合
keys_from_csv1 = set()
for path in df1['file_path'].dropna():
    key = extract_key_from_file_path(path)
    if key:
        keys_from_csv1.add(key)

# 为 csv2 每行生成 key，并过滤
filtered_rows = []
for _, row in df2.iterrows():
    vid_path = row['video_url']
    if pd.isna(vid_path):
        continue
    key = extract_key_from_vid_path(vid_path)
    if key and key in keys_from_csv1:
        filtered_rows.append(row)

# 构建新 DataFrame
df2_filtered = pd.DataFrame(filtered_rows, columns=df2.columns)

# 保存结果（可选覆盖原文件或另存）
output_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/0919_openhuman_one_human_all_25fps_part_1_final_filtered.csv"
df2_filtered.to_csv(output_path, index=False)

print(f"Original csv2 rows: {len(df2)}")
print(f"Filtered csv2 rows: {len(df2_filtered)}")
print(f"Saved to: {output_path}")