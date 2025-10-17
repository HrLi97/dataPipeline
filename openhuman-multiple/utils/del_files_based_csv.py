import pandas as pd
import os
import glob

# ———— 参数区 ————
yuan_csv = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/out_4_22_final_2-filtered_dedup.csv"
depth    = 1  # 仅对从 video_path 向上走 depth 级所到的目录做 mp4 清理
is_del = True

df = pd.read_csv(yuan_csv)
# df = df.drop_duplicates(subset=["video_path", "audio_path", "mask_jsonl", "pose_jsonl"])

video_paths = set(df["video_path"].dropna().tolist())
audio_paths = set(df["audio_path"].dropna().tolist())
mask_paths  = set(df["mask_jsonl"].dropna().tolist())
pose_paths  = set(df["pose_jsonl"].dropna().tolist())
visualize_tracks_paths  = set(df["visualize_tracks"].dropna().tolist())


print(f"Unique entries in CSV → videos: {len(video_paths)}, audio: {len(audio_paths)}, masks: {len(mask_paths)}, poses: {len(pose_paths)}")

deleted = {"video":0, "audio":0, "mask":0, "pose":0, "tracks":0}
kept    = {"video":0, "audio":0, "mask":0, "pose":0, "tracks":0}


target_dirs = {
    os.path.sep.join(video_path.split(os.path.sep)[:-depth])  # 去掉末尾 depth 个目录层级
    for video_path in video_paths
}

for td in sorted(target_dirs):
    if not os.path.isdir(td):
        print(f"⚠️ skipped (not dir): {td}")
        continue
    for fname in os.listdir(td):
        if not fname.lower().endswith(".mp4"):
            continue
        fp = os.path.join(td, fname)
        if fp in video_paths:
            kept["video"] += 1
        else:
            if is_del:
                os.remove(fp)
            deleted["video"] += 1

target_dirs_tracks = {
    os.path.sep.join(video_path.split(os.path.sep)[:-depth])  # 去掉末尾 depth 个目录层级
    for video_path in visualize_tracks_paths
}

print(target_dirs_tracks,"target_dirs_trackstarget_dirs_tracks")

for td in sorted(target_dirs_tracks):
    if not os.path.isdir(td):
        print(f"⚠️ skipped (not dir): {td}")
        continue
    for fname in os.listdir(td):
        if not fname.lower().endswith(".mp4"):
            continue
        fp = os.path.join(td, fname)
        if fp in visualize_tracks_paths:
            kept["tracks"] += 1
        else:
            if is_del:
                os.remove(fp)
            deleted["tracks"] += 1

# —— 4. WAV 清理 —— #
# 找到所有 audio_paths 的公共根，再递归删除未在列表里的 .wav
if audio_paths:
    root_audio = os.path.commonpath(list(audio_paths))
    for fp in glob.glob(os.path.join(root_audio, "**", "*.wav"), recursive=True):
        if fp in audio_paths:
            kept["audio"] += 1
        else:
            if is_del:
                os.remove(fp)
            deleted["audio"] += 1

# —— 5. MASK JSONL 清理 —— #
if mask_paths:
    root_mask = os.path.commonpath(list(mask_paths))
    for fp in glob.glob(os.path.join(root_mask, "**", "*_mask.jsonl"), recursive=True):
        if fp in mask_paths:
            kept["mask"] += 1
        else:
            if is_del:
                os.remove(fp)
            deleted["mask"] += 1

# —— 6. POSE JSONL 清理 —— #
if pose_paths:
    root_pose = os.path.commonpath(list(pose_paths))
    for fp in glob.glob(os.path.join(root_pose, "**", "*_landmarks.jsonl"), recursive=True):
        if fp in pose_paths:
            kept["pose"] += 1
        else:
            if is_del:
                os.remove(fp)
            deleted["pose"] += 1

# —— 7. 打印汇总 —— #
print("\nCleanup summary:")
print(f"  Videos → kept: {kept['video']}, deleted: {deleted['video']}")
print(f"  Audio  → kept: {kept['audio']}, deleted: {deleted['audio']}")
print(f"  Masks  → kept: {kept['mask']}, deleted: {deleted['mask']}")
print(f"  Poses  → kept: {kept['pose']}, deleted: {deleted['pose']}")
print(f"  tracks  → kept: {kept['tracks']}, deleted: {deleted['tracks']}")
