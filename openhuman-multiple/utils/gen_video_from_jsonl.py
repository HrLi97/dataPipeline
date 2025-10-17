import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils

video_path     = "/home/ubuntu/MyFiles/haoran/code/data_source_all/tvshow/batch-1/TV_Series/RGZC-01/RGZC-01_114-117.mp4"
landmark_file  = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/2-out-new/saved_vid_root/jsonl/RGZC-01_114-117_landmarks.jsonl"
mask_file      = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/2-out-new/saved_vid_root/jsonl/RGZC-01_114-117_mask.jsonl"
output_path    = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/utils/RGZC-01_114-117_vis.mp4"

landmarks = {}
with open(landmark_file, 'r') as f:
    for frame_idx, line in enumerate(f):
        obj = json.loads(line)
        # obj: dict of instance_id -> {"keypoints":[…]}
        landmarks[frame_idx] = obj  # 整个 dict

# ———— 3. 读取并解码 masks ————
# 每行 JSON 结构类似：{"85": {"size":[H,W], "counts":"…"}, "86":{…}, …}
masks = {}
with open(mask_file, 'r') as f:
    for frame_idx, line in enumerate(f):
        obj = json.loads(line)
        rle_list = []
        for inst_id, coco_rle in obj.items():
            rle = {
                "size": tuple(coco_rle["size"]),
                "counts": coco_rle["counts"].encode("utf-8"),
            }
            m = maskUtils.decode(rle)  # H×W 二值 mask
            rle_list.append(m)
        masks[frame_idx] = rle_list

# ———— 4. 打开视频，准备写入 ————
cap = cv2.VideoCapture(video_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ———— 5. 逐帧叠加并写出 ————
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    vis = frame.copy()

    # 叠加所有 mask：半透明绿色
    if frame_idx in masks:
        for m in masks[frame_idx]:
            mask_bool = m.astype(bool)
            # 对 mask 区域做 0.6*原图 + 0.4*绿色
            green = np.zeros_like(vis)
            green[...,1] = 255
            vis[mask_bool] = cv2.addWeighted(vis, 0.6, green, 0.4, 0)[mask_bool]

    # 绘制所有 instance 的 keypoints：红点
    if frame_idx in landmarks:
        for inst_id, inst_data in landmarks[frame_idx].items():
            for x, y in inst_data["keypoints"]:
                cv2.circle(vis, (int(x), int(y)), radius=3, color=(0,0,255), thickness=-1)

    out.write(vis)
    frame_idx += 1

cap.release()
out.release()
print(f"可视化视频已保存到 {output_path}")