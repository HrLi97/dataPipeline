import os
import csv

# —— 配置 —— 
csv_path = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/FLN_metadata.csv"

visual_root   = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/visualization"
jsonl_root    = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/jsonl"
whisper_root  = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/whisper_feat"
video_root    = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/FLN/data/saved_vid_root/yuan/TV_Series"

# 1. 读取 CSV，收集“要保留”的完全路径
keep = set()
count = 0
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for col in ("visual_path", "landmarks_path", "mask_path", "npy_path", "file_path"):
            p = row.get(col, "").strip()
            if p:
                if p.endswith(","):
                    p = p[:-1]
                keep.add(p)
                count+=1
                print(p,"ppppppppppppppppppppppppppppppp")

print(count,"countcountcount")

# 2. 定义一个函数：遍历某个根目录，删除不在 keep 中的文件
def clean_dir(root):
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if full not in keep:
                try:
                    os.remove(full)
                    print(f"Removed: {full}")
                except Exception as e:
                    print(f"Failed to remove {full}: {e}")

print("Cleaning visual files...")
clean_dir(visual_root)

print("Cleaning landmarks & mask JSONL...")
clean_dir(jsonl_root)

print("Cleaning whisper npy files...")
clean_dir(whisper_root)

print("Cleaning original video files...")
clean_dir(video_root)

print("Done.")
