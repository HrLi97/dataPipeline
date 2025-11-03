# -- coding: utf-8 --
import json
import logging
import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any, List
import torch
from PIL import Image
from decord import VideoReader, cpu

# ====== 环境 & 路径设置 ======
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 插入必要路径（Ray 会通过 runtime_env 传递，本地也需保证）
sys.path.insert(0, "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/mmpose-main")
sys.path.insert(0, "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/Music-Source-Separation-Training")
sys.path.insert(0, "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/syncnet_python-master")
sys.path.append("/mnt/cfs/shanhai/lihaoran/Data_process/a6000")
sys.path.append("/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/")
sys.path.append("/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/grounding_dino/")
sys.path.append("/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/sam2_opt/sam2")

# ====== 全局配置 ======
GROUNDING_DINO_CONFIG = "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/ckps/groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.4
SAMPLE_INTERVAL = 5
MIN_RATIO = 0.9  # 90% 的采样帧必须有 2 人

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

_grounding_model = None

def get_grounding_model(device="cuda"):
    global _grounding_model
    if _grounding_model is None:
        print("Loading GroundingDINO model...")
        _grounding_model = load_model(
            GROUNDING_DINO_CONFIG,
            GROUNDING_DINO_CHECKPOINT,
            device=device
        )
    return _grounding_model

# ====== 核心判断函数 ======
def is_two_person_video(vid_path: str) -> bool:
    try:
        vr = VideoReader(vid_path, ctx=cpu(0))
        total_frames = len(vr)
        if total_frames == 0:
            return False

        sample_frames = list(range(0, total_frames, SAMPLE_INTERVAL))
        two_person_count = 0

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        model = get_grounding_model()

        for frame_idx in sample_frames:
            frame_np = vr[frame_idx].asnumpy()
            image_pil = Image.fromarray(frame_np).convert("RGB")
            image_transformed, _ = transform(image_pil, None)

            boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,
                caption="face.",
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            if len(boxes) == 2:
                two_person_count += 1

        ratio = two_person_count / len(sample_frames) if sample_frames else 0
        return ratio >= MIN_RATIO
    except Exception as e:
        print(f"⚠️ Error processing {vid_path}: {e}")
        return False

class Worker:
    def __call__(self, row):
        vid_path = row["path"]
        is_two = is_two_person_video(vid_path)
        row["is_two_person"] = is_two
        return row

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_path", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/data/open_human_all/all_csv_local/OpenHumanVid_part_004.sample10.csv", help="Input CSV with 'path' column")
    parser.add_argument("--ray_log_dir", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/two_person/filter/part-004/", help="Output dir for Ray (e.g., ./output/)")
    parser.add_argument("--log_path", default="/tmp/two_person_filter.log", help="Log file path")
    parser.add_argument("--is_local", type=bool, default=False, help="Run in local multi-GPU mode")
    parser.add_argument("--disable_detailed_pbar", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()

# ====== 主函数 ======
def main():
    opt = parse_args()

    os.makedirs(os.path.dirname(opt.log_path), exist_ok=True)
    logging.basicConfig(
        filename=opt.log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("🚀 Two-person filter started with input: %s", opt.input_csv_path)

    os.makedirs(opt.ray_log_dir, exist_ok=True)

    if opt.is_local:
        from accelerate import PartialState
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)

        worker = Worker()
        with distributed_state.split_between_processes(samples, apply_padding=True) as batch:
            for item in batch:
                result = worker([item])[0]
                print(f"[Local] {item['path']} → is_two_person: {result['is_two_person']}")
                
    if opt.is_local:
        from accelerate import PartialState
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        torch.cuda.set_device(device_id)

        worker = Worker()
        all_results = []
        with distributed_state.split_between_processes(samples, apply_padding=True) as batch:
            for item in batch:
                result = worker([item])[0]
                print(f"[Local] {item['path']} → is_two_person: {result['is_two_person']}")
                all_results.append(result)

        # Gather results from all processes
        all_results = distributed_state.gather_object(all_results)

        if distributed_state.is_main_process:
            # Only keep True entries
            filtered = [r for r in all_results if r.get("is_two_person") is True]
            output_path = os.join(opt.ray_log_dir,"local.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if filtered:
                with open(output_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
                    writer.writeheader()
                    writer.writerows(filtered)
            
    else:
        import ray
        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "PYTHONPATH": ":".join([
                        "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/mmpose-main",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/Music-Source-Separation-Training",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/syncnet_python-master",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/sam2_opt",
                    ])
                }
            }
        )

        ds = ray.data.read_csv(opt.input_csv_path)
        result_ds = ds.map(
            Worker,
            num_gpus=0.5, 
            concurrency=2,
        )
        result_ds.write_csv(opt.ray_log_dir)
        print(f"✅ Ray job finished. Results saved to: {opt.ray_log_dir}")

if __name__ == "__main__":
    main()