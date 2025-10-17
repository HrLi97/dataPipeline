import os
import pandas as pd

# 配置路径
base_csv = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/out_4_22_final_2-filtered_dedup.csv"
npy_root  = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/out_4_22_final_2/saved_vid_root/whisper_feat"

df = pd.read_csv(base_csv)

npy_map = {}
for root, _, files in os.walk(npy_root):
    for fn in files:
        if fn.lower().endswith(".npy"):
            stem = os.path.splitext(fn)[0]
            npy_map[stem] = os.path.join(root, fn)

def lookup_npy(row):
    wav_path = row.get("audio_path", "")
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    return npy_map.get(stem, "")  # 若未找到，则返回空字符串

# 4. 应用函数，新增一列
df["audio_npy_path"] = df.apply(lookup_npy, axis=1)

# （可选）查看哪些行没找到对应的 npy
missing = df["audio_npy_path"] == ""
print(f"未找到 .npy 的行数：{missing.sum()} / {len(df)}")

# 5. 保存回 CSV（覆盖原文件或另存）
df.to_csv(base_csv, index=False)
print(f"已在 `{base_csv}` 中新增列 `audio_npy_path`。")
