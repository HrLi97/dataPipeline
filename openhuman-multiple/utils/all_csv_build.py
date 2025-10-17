import os
import csv

# —— 配置各目录 —— 
file_dir  = "/home/ubuntu/MyFiles/haoran/code/data_source_all/tvshow/batch-2/FLN/TV_Series"
jsonl_dir  = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/jsonl"
npy_dir    = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/whisper_feat/"
visual_dir = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/visualization"
moss_dir   = "/home/ubuntu/MyFiles/haoran/code/data_source_all/tvshow/batch-2/FLN/TV_Series"
output_csv = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/FLN_metadata111.csv"

fieldnames = ["file_path", "landmarks_path", "mask_path", "npy_path", "moss_path","visual_path"]
with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for fname in sorted(os.listdir(visual_dir)):
        if not fname.endswith(".mp4"):
            continue

        base = os.path.splitext(fname)[0].split("_vis")[0]  # e.g. "FLN-01_265-269"
        print(base,"basebasebasebase")
        video_path = os.path.join(file_dir, fname)
        landmarks_path = os.path.join(jsonl_dir, f"{base}_landmarks.jsonl")
        mask_path = os.path.join(jsonl_dir, f"{base}_mask.jsonl")
        
        moss_filename = f"{base}.mp4"
        moss_path = ""
        for root, dirs, files in os.walk(moss_dir):
            if moss_filename in files:
                moss_path = os.path.join(root, moss_filename)
                break

        npy_filename = f"{base}_SE48K.npy"
        npy_path = ""
        for root, dirs, files in os.walk(npy_dir):
            if npy_filename in files:
                npy_path = os.path.join(root, npy_filename)
                break

        # —— 只有当所有文件都存在时，才写这一行 —— 
        if all(os.path.isfile(p) for p in (landmarks_path, mask_path, npy_path, moss_path)):
            writer.writerow({
                "file_path":video_path,
                "landmarks_path": landmarks_path,
                "mask_path": mask_path,
                "npy_path": npy_path,
                "moss_path": moss_path,
            })
        else:
            # 可以选着打印哪些基准缺失以便调试
            missing = [label for label, p in [
                ("landmarks", landmarks_path),
                ("mask",      mask_path),
                ("npy",       npy_path),
                ("moss",      moss_path),
            ] if not os.path.isfile(p)]
            print(f"[WARN] 跳过 {video_path}：缺失 {missing}")
