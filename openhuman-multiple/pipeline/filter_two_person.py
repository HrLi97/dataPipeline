# -*- coding: utf-8 -*-
import glob
import os, csv, argparse, logging, math, subprocess
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

try:
    import ffmpeg
    _USE_FFMPEG_PY = True
except Exception:
    _USE_FFMPEG_PY = False
    
'''
export CUDNN_LIB_DIR="$(python -c 'import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))')"
export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
'''



# # # ---- 放在最顶部，任何 TensorFlow 导入之前 ----
# import os
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true") 
# os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0") 

# import tensorflow as tf
# try:
#     gpus = tf.config.list_physical_devices('GPU')
#     for g in gpus:
#         tf.config.experimental.set_memory_growth(g, True)
# except Exception:
#     pass


# import os
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "") 
# import tensorflow as tf
# try:
#     tf.config.set_visible_devices([], 'GPU')
# except Exception:
#     pass


from retinaface import RetinaFace

def load_skip_set(specs) -> set:

    skip = set()
    if not specs:
        return skip

    def _iter_rows_from_csv(csv_path: str):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r.get("loca_path_a100")).strip()
                if key:
                    yield key

    try:
        cand = expand_input_csvs([specs])
        if len(cand) == 1:
            try:
                with open(cand[0], "r", encoding="utf-8-sig") as f:
                    peek = csv.DictReader(f)
                    if "loca_path_a100" in (peek.fieldnames or []):
                        leafs = []
                        for r in peek:
                            p = (r.get("loca_path_a100") or "").strip()
                            if p:
                                leafs.extend(expand_input_csvs([p]))
                        for lp in sorted(set(leafs)):
                            for k in _iter_rows_from_csv(lp):
                                skip.add(k)
                    else:
                        for k in _iter_rows_from_csv(cand[0]):
                            skip.add(k)
            except Exception:
                for p in cand:
                    for k in _iter_rows_from_csv(p):
                        skip.add(k)
        else:
            for p in cand:
                for k in _iter_rows_from_csv(p):
                    skip.add(k)
    except FileNotFoundError:
        pass

    logging.warning(f"[already_over_csvs] loaded {len(skip)} items to skip.")
    return skip


def expand_input_csvs(specs) -> List[str]:
    out = []
    cand = []
    for s in specs:
        if not s:
            continue
        # 允许在参数里写逗号分隔
        cand.extend([x.strip() for x in s.split(",") if x.strip()])

    for s in cand:
        if os.path.isdir(s):
            out.extend(sorted(glob.glob(os.path.join(s, "*.csv"))))
        elif any(ch in s for ch in "*?[]"):
            out.extend(sorted(glob.glob(s)))
        elif os.path.isfile(s):
            out.append(s)
        else:
            raise FileNotFoundError(f"找不到输入：{s}")
    # 去重
    return sorted(set(out))

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)

def write_csv(path: str, rows: List[Dict[str, Any]]):
    ensure_dir(path)
    if not rows:
        # 空表头兜底
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(["video_path","status"])
        return
    # 统一字段（原字段 + 计算字段）
    keys = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = sorted(keys)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def has_audio_stream(video_path: str) -> bool:
    # 先用 ffmpeg-python
    if _USE_FFMPEG_PY:
        info = ffmpeg.probe(video_path)
        streams = info.get("streams", []) or []
        return any(s.get("codec_type") == "audio" for s in streams)

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", video_path
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    return out.returncode == 0 and out.stdout.strip() != ""

# ------------------------- RetinaFace 解析 -------------------------
def _extract_face_boxes_from_retinaface(ret_res) -> List[Tuple[int,int,int,int]]:

    boxes = []
    if ret_res is None:
        return boxes
    if isinstance(ret_res, dict):
        for _k, v in ret_res.items():
            if isinstance(v, dict) and "facial_area" in v:
                fa = v["facial_area"]
                if isinstance(fa, (list, tuple)) and len(fa) == 4:
                    x1,y1,x2,y2 = fa
                    boxes.append((int(x1), int(y1), int(x2), int(y2)))
    elif isinstance(ret_res, (list, tuple)):
        for v in ret_res:
            if isinstance(v, dict) and "facial_area" in v and len(v["facial_area"]) == 4:
                x1,y1,x2,y2 = v["facial_area"]
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


# ------------------------- 人脸面积统计（整帧） -------------------------
class FaceAreaJudgeRetinaFull:
   
    def __init__(self, frames_to_sample: int = 24, frame_stride: int = 0):
        self.frames_to_sample = frames_to_sample
        self.frame_stride = frame_stride

    def _sample_frames(self, cap: cv2.VideoCapture, total: int) -> List[np.ndarray]:
        frames = []
        if total <= 0:
            return frames
        stride = self.frame_stride if self.frame_stride > 0 else max(1, total // max(1, self.frames_to_sample))
        idxs = list(range(0, total, stride))[:self.frames_to_sample]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                frames.append(bgr)
        return frames

    def face_area_ratio_on_frame(self, bgr: np.ndarray) -> float:
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return 0.0
        faces = RetinaFace.detect_faces(bgr)
        boxes = _extract_face_boxes_from_retinaface(faces)
        if not boxes:
            return 0.0
        frame_area = float(h * w)
        ratios = []
        for (x1,y1,x2,y2) in boxes:
            fw = max(0, x2 - x1)
            fh = max(0, y2 - y1)
            if fw == 0 or fh == 0:
                continue
            ratios.append((fw * fh) / frame_area)
        return max(ratios) if ratios else 0.0

    def summarize_video(self, video_path: str) -> Dict[str, float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"face_ratio_max": 0.0, "face_ratio_mean": 0.0, "frames_used": 0}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = self._sample_frames(cap, total)
        ratios = [self.face_area_ratio_on_frame(bgr) for bgr in frames]
        cap.release()
        if not ratios:
            return {"face_ratio_max": 0.0, "face_ratio_mean": 0.0, "frames_used": 0}
        return {
            "face_ratio_max": float(np.max(ratios)),
            "face_ratio_mean": float(np.mean(ratios)),
            "frames_used": len(ratios),
        }
        
import subprocess
def probe_resolution(video_path: str, timeout: int = 5) -> Tuple[Optional[int], Optional[int]]:
    # try:
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w > 0 and h > 0:
            return w, h
    # except Exception:
    #     pass
    return None, None


def process_one(row: Dict[str, Any], args, judge: FaceAreaJudgeRetinaFull) -> Dict[str, Any]:

    rec: Dict[str, Any] = {}
    video_path = (row.get("video_path") or row.get("path") or row.get("video") or "").strip()
    rec["video_path"] = video_path

    for k in ("id", "uid", "sha", "title", "aesthetic"):
        if k in row:
            rec[k] = row[k]

    w, h = probe_resolution(video_path)
    rec["width"], rec["height"] = w, h
    if not (w and h):
        rec["status"] = "res_probe_fail"
        rec["error"] = "cannot_probe_resolution"
        return rec
    if not (w > 640 and h > 640):
        rec["status"] = "small_resolution"
        return rec

    # 就只需要判断音频 质量 人脸占比
    # 1) 音频
    has_audio = has_audio_stream(video_path)
    rec["has_audio"] = int(has_audio)
    if args.require_audio and not has_audio:
        rec["status"] = "no_audio"
        return rec

    # 2) aesthetic
    aest = safe_float(row.get("aesthetic", np.nan))
    rec["aesthetic"] = aest
    if not math.isfinite(aest) or aest < args.aesthetic_threshold:
        rec["status"] = "low_aesthetic"
        return rec

    # 3) 人脸占比（整帧 RetinaFace）
    stats = judge.summarize_video(video_path)
    rec.update(stats)
    if stats["face_ratio_max"] < args.face_area_threshold:
        rec["status"] = "small_face"
        return rec

    rec["status"] = "valid"
    return rec

parser = argparse.ArgumentParser(description="Multi-CSV filter with RetinaFace on full frame; outputs ONE merged CSV.")
parser.add_argument("--already_over_csvs", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/data/open_human_all/0901/0901-merged_with_a100.csv")
parser.add_argument("--input_csvs", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/data/open_human_all/part030609.csv")
parser.add_argument("--output_csv_path", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/data/open_human_all/filter/part1-10.csv")
parser.add_argument("--aesthetic_threshold", type=float, default=5.2)
parser.add_argument("--face_area_threshold", type=float, default=0.02)
parser.add_argument("--require_audio", default=True, help="开启后：无音频直接拒绝")
parser.add_argument("--frames_to_sample", type=int, default=100)
parser.add_argument("--frame_stride", type=int, default=10, help=">0 使用固定步长抽帧；否则自适应均匀采样")
parser.add_argument('--is_local', type=bool, default=True)
parser.add_argument("--log_path", default=None)
parser.add_argument("--only_valid", default=True)
args = parser.parse_args()
SKIP_SET = frozenset(load_skip_set(args.already_over_csvs)) 
print(SKIP_SET,"SKIP_SETSKIP_SETSKIP_SET")
logging.basicConfig(
    filename=args.log_path,
    level=logging.INFO if args.log_path else logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# FIELDNAMES = ["video_path","aesthetic","width","height","has_audio",
#     "face_ratio_max","face_ratio_mean","frames_used","status","error"
# ]

# def sanitize(rec: dict) -> dict:
#     return {k: ("" if rec.get(k) is None else rec.get(k, "")) for k in FIELDNAMES}

FIELDNAMES = [
    "video_path","aesthetic","width","height","has_audio",
    "face_ratio_max","face_ratio_mean","frames_used","status","error"
]

NUM_FLOAT = ["aesthetic","face_ratio_max","face_ratio_mean"]
NUM_INT   = ["width","height","frames_used","has_audio"]
STR_COLS  = ["video_path","status","error"]

def _to_float(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip() == "": return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def _to_int(x):
    try:
        if x is None: return None
        if isinstance(x, str) and x.strip() == "": return None
        return int(float(x))
    except Exception:
        return None

def sanitize(rec: dict) -> dict:
    out = {k: rec.get(k) for k in FIELDNAMES}

    for c in NUM_FLOAT:
        out[c] = _to_float(out.get(c))
    for c in NUM_INT:
        out[c] = _to_int(out.get(c))
    for c in STR_COLS:
        v = out.get(c)
        out[c] = "" if v is None else str(v)

    return out

def process_row(row: dict) -> dict:
    # try:
    rec = process_one(row, args, FaceAreaJudgeRetinaFull(args.frames_to_sample, args.frame_stride))
    # except Exception as e:
    #     rec = {
    #         "video_path": (row.get("video_path") or row.get("path") or row.get("video") or "").strip(),
    #         "status": "error",
    #         "error": f"exception:{type(e).__name__}:{e}",
    #     }
    return sanitize(rec)

if __name__ == "__main__":
    
    if args.is_local:
        from tqdm import tqdm

        def _iter_rows_from_csv(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    yield r

        candidates = expand_input_csvs([args.input_csvs])
        input_csvs = []

        if len(candidates) == 1:
            # 判断是否是“清单CSV（含 file_path 列）”
            try:
                with open(candidates[0], "r", encoding="utf-8-sig") as f:
                    peek = csv.DictReader(f)
                    if "file_path" in (peek.fieldnames or []):
                        leafs = []
                        for r in peek:
                            p = (r.get("file_path") or "").strip()
                            if p:
                                leafs.extend(expand_input_csvs([p]))
                        input_csvs = sorted(set(leafs))
                    else:
                        input_csvs = candidates
            except Exception:
                input_csvs = candidates
        else:
            input_csvs = candidates

        seen = set()
        judge = FaceAreaJudgeRetinaFull(args.frames_to_sample, args.frame_stride)
        ensure_dir(args.output_csv_path)
        
        with open(args.output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()

            # 逐个 CSV、逐行遍历，边算边写
            for p in input_csvs:
                with open(p, "r", encoding="utf-8-sig") as fin:
                    reader = csv.DictReader(fin)
                    for row in reader:
                        key = (row.get("video_path") or row.get("path") or row.get("video") or "").strip()
                        # 跳过无效/重复/已处理
                        if (not key) or (key in seen) or (key in SKIP_SET):
                            continue
                        seen.add(key)

                        # try:
                        rec = process_one(row, args, judge)
                        print(rec,"recrecrecrec")
                        # except Exception as e:
                        #     rec = {
                        #         "video_path": key,
                        #         "status": "error",
                        #         "error": f"exception:{type(e).__name__}:{e}",
                        #     }
        
                        if args.only_valid:
                            if rec.get("status") == "valid":
                                w.writerow(sanitize(rec))
                        else:
                            w.writerow(sanitize(rec))

    else:
        import ray
        import shutil, glob, os
        import pandas as pd
        import pyarrow as pa
        paths = pd.read_csv(args.input_csvs, dtype=str)["file_path"].dropna().tolist()

        # samples = ray.data.read_csv(paths)
        samples = ray.data.read_csv(paths) 
        print("samples.count",samples.count())
        # samples = samples.repartition(samples.count())
        
        def process_row_with_skip(row: dict) -> dict:
            key = (row.get("video_path") or row.get("path") or row.get("video") or "").strip()
            if key in SKIP_SET:
                return sanitize({
                    "video_path": key,
                    "status": "skipped_already",
                })
            return process_row(row)

        # samples = ray.data.read_csv(args.input_csvs)
        predictions = samples.map(
            process_row_with_skip,
            num_gpus=1,
            concurrency=6,
        )
        ds = predictions.filter(lambda r: r.get("status") == "valid")
        # 最后合并
        # ds = ds.repartition(1)
        # 一步一步写
        out_dir = os.path.splitext(args.output_csv_path)[0] + "_shards"
        shutil.rmtree(out_dir, ignore_errors=True)
        ds.write_csv(out_dir)

        parts = sorted(glob.glob(os.path.join(out_dir, "part-*.csv")))
        with open(args.output_csv_path, "wb") as w:
            for i, p in enumerate(parts):
                with open(p, "rb") as r:
                    if i > 0:
                        r.readline()
                    shutil.copyfileobj(r, w)

        # shutil.rmtree(out_dir)
        out_dir = os.path.dirname(args.output_csv_path)