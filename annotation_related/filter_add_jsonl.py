import pandas as pd

moss_csv_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/part-1-moss-after.csv"
pipeline_csv_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/fps/open_part/0919_openhuman_one_human_all_25fps_part_1.csv"
output_csv = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/part-1-OK.csv"
part_num = 1

df_moss = pd.read_csv(moss_csv_path)
df_pipe = pd.read_csv(pipeline_csv_path)

# 1. 从 moss.csv 的 file_path 中提取 video_id，并保留映射
def extract_id_from_moss_path(path):
    basename = path.split("/")[-1]
    return basename.split("__")[0]

df_moss["video_id"] = df_moss["file_path"].apply(extract_id_from_moss_path)

# 构建 video_id -> file_path 的映射（假设每个 ID 唯一）
moss_id_to_vis = dict(zip(df_moss["video_id"], df_moss["file_path"]))
moss_ids = set(moss_id_to_vis.keys())

# 2. 从 pipeline CSV 提取 video_id
def extract_id_from_pipe_url(url):
    basename = url.split("/")[-1]
    return basename.split("__")[0]

df_pipe["video_id"] = df_pipe["video_url"].apply(extract_id_from_pipe_url)

# 3. 筛选
df_filtered = df_pipe[df_pipe["video_id"].isin(moss_ids)].copy()

# 4. 添加 pose_jsonl 和 vis_path
def build_jsonl_url(video_id):
    prefix = video_id[:2]
    return f"http://10.1.200.150/shanhai/lihaoran/data/annotation/openhuman/part-{part_num}/{prefix}/{video_id}__fps25_wholebody.jsonl"

df_filtered["pose_jsonl"] = df_filtered["video_id"].apply(build_jsonl_url)
df_filtered["vis_path"] = df_filtered["video_id"].map(moss_id_to_vis)

# 5. 删除临时列
df_filtered = df_filtered.drop(columns=["video_id"])

# 6. 保存
df_filtered.to_csv(output_csv, index=False)
print(f"Saved {len(df_filtered)} rows to {output_csv}")