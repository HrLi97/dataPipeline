#!/usr/bin/env python3
import csv
import json
import os
import argparse
from pathlib import Path
import subprocess
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, unquote
import shutil
import mimetypes
import pandas as pd
import numpy as np
import ray
from datetime import datetime, date

# ------------------ Constants ------------------
FFMPEG_COMMON_FLAGS = ["-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
FPS_DEFAULT = 25.0
FPS_TOL_DEFAULT = 0.05


# ------------------ Utility Functions ------------------
def _csv_safe_scalar(v):
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (list, tuple, set)):
        try:
            return json.dumps(list(v), ensure_ascii=False)
        except Exception:
            return str(v)
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)


def _csv_sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _csv_safe_scalar(v) for k, v in row.items()}


def _parse_ffprobe_rate(rate: str) -> Optional[float]:
    if not rate or rate in ("0/0", "N/A"):
        return None
    rate = rate.strip()
    try:
        if "/" in rate:
            n, d = rate.split("/", 1)
            n, d = float(n), float(d)
            return n / d if d != 0 else None
        return float(rate)
    except Exception:
        return None


def probe_fps(path: str) -> Optional[float]:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found in PATH")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    # Try JSON method first
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            video_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
            if video_streams:
                stream = next((s for s in video_streams if (s.get("disposition") or {}).get("default")), video_streams[0])
                fps = _parse_ffprobe_rate(stream.get("avg_frame_rate")) or _parse_ffprobe_rate(stream.get("r_frame_rate"))
                if fps and fps > 0:
                    return fps
    except Exception:
        pass

    # Fallback to direct key extraction
    for key in ("avg_frame_rate", "r_frame_rate"):
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", f"stream={key}",
            "-of", "default=nokey=1:noprint_wrappers=1", path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            fps = _parse_ffprobe_rate(result.stdout.strip())
            if fps and fps > 0:
                return fps
        except Exception:
            continue

    return None


def _get_local_path(item: Dict[str, Any]) -> Optional[str]:
    cand = item.get('path')
    if isinstance(cand, (list, tuple)):
        cand = cand[0]
    if not cand:
        return None
    p = urlparse(str(cand))
    if p.scheme in ("http", "https"):
        return str(cand)
    if p.scheme == "file":
        return unquote(p.path)
    return str(cand)


def _mk_out_path(src_root: str, out_root: str, src_path: str, fps_required: float) -> Path:
    src_root = Path(src_root).resolve()
    out_root = Path(out_root)
    src_path = Path(src_path).resolve()
    rel_path = src_path.relative_to(src_root)
    out_dir = out_root / rel_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = rel_path.stem if rel_path.suffix else rel_path.name
    fps_int = int(round(fps_required))
    out_name = f"{stem}__fps{fps_int}.mp4"
    return out_dir / out_name


def process_to_dir(
    input_path: str,
    out_root: str,
    src_root: str,
    fps_required: float = FPS_DEFAULT,
    fps_tol: float = FPS_TOL_DEFAULT,
    overwrite: bool = False,
    copy_if_same_fps: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"file_not_found: {input_path}")

    out_path = _mk_out_path(src_root, out_root, input_path, fps_required)
    if os.path.exists(out_path) and not overwrite:
        fps_measured = probe_fps(out_path)
        return {"out_path": str(out_path), "measured_fps": fps_measured, "transcode": False}

    out_tmp = str(out_path) + ".part"
    if os.path.exists(out_tmp):
        os.remove(out_tmp)

    fps = probe_fps(input_path)
    need_transcode = not (fps is not None and abs(fps - fps_required) <= fps_tol)

    try:
        if need_transcode:
            base_cmd = [
                "ffmpeg", *FFMPEG_COMMON_FLAGS,
                "-i", input_path,
                "-vf", f"fps={int(fps_required)}",
                "-vsync", "cfr",
                "-f", "mp4",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "10",
                "-c:a", "copy",
                "-movflags", "+faststart",
                out_tmp,
            ]
            try:
                subprocess.run(base_cmd, check=True, timeout=600)
            except subprocess.CalledProcessError:
                fallback_cmd = [
                    "ffmpeg", *FFMPEG_COMMON_FLAGS,
                    "-i", input_path,
                    "-vf", f"fps={int(fps_required)}",
                    "-vsync", "cfr",
                    "-f", "mp4",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "10",
                    "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart",
                    out_tmp,
                ]
                subprocess.run(fallback_cmd, check=True, timeout=600)
        else:
            if copy_if_same_fps:
                copy_cmd = [
                    "ffmpeg", *FFMPEG_COMMON_FLAGS,
                    "-i", input_path,
                    "-f", "mp4",
                    "-c", "copy",
                    "-movflags", "+faststart",
                    out_tmp,
                ]
                try:
                    subprocess.run(copy_cmd, check=True, timeout=600)
                except subprocess.CalledProcessError:
                    reencode_cmd = [
                        "ffmpeg", *FFMPEG_COMMON_FLAGS,
                        "-i", input_path,
                        "-f", "mp4",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "10",
                        "-c:a", "aac", "-b:a", "160k",
                        "-movflags", "+faststart",
                        out_tmp,
                    ]
                    subprocess.run(reencode_cmd, check=True, timeout=600)
                    need_transcode = True
            else:
                force_cmd = [
                    "ffmpeg", *FFMPEG_COMMON_FLAGS,
                    "-i", input_path,
                    "-vf", f"fps={int(fps_required)}",
                    "-vsync", "cfr",
                    "-f", "mp4",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "10",
                    "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart",
                    out_tmp,
                ]
                subprocess.run(force_cmd, check=True, timeout=600)
                need_transcode = True

        os.replace(out_tmp, out_path)
        return {"out_path": str(out_path), "measured_fps": fps, "transcode": need_transcode}

    finally:
        if os.path.exists(out_tmp):
            try:
                os.remove(out_tmp)
            except Exception:
                pass


# ------------------ Worker Class ------------------
class Worker:
    def __init__(self, args_dict: Dict[str, Any]):
        self.args = args_dict

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(item)
        for key in ['proc_status', 'proc_path_fps25', 'proc_transcode', 'proc_measured_fps', 'proc_error']:
            result.setdefault(key, None)

        path = _get_local_path(result)
        if not path:
            result.update({
                'proc_error': 'no_path_field_found',
                'proc_status': 0
            })
            return result

        if urlparse(path).scheme in ("http", "https"):
            result.update({
                'proc_error': 'remote_http_https_not_supported',
                'proc_status': 0
            })
            return result

        try:
            info = process_to_dir(
                input_path=path,
                out_root=self.args['out_root'],
                src_root=self.args['src_root'],
                fps_required=self.args['fps_required'],
                fps_tol=self.args['fps_tol'],
                overwrite=self.args['overwrite'],
                copy_if_same_fps=not self.args['no_copy_if_same'],
            )
            result.update({
                'proc_status': 1,
                'proc_path_fps25': info['out_path'],
                'proc_measured_fps': info['measured_fps'],
                'proc_transcode': int(info['transcode']),
                'proc_error': None
            })
        except Exception as e:
            result.update({
                'proc_status': 0,
                'proc_error': str(e)
            })
        return result


# ------------------ CLI & Main ------------------
def validate_args(args):
    if args.fps_required <= 0:
        raise ValueError("--fps_required must be positive")
    if args.fps_tol < 0:
        raise ValueError("--fps_tol must be non-negative")
    if not os.path.isfile(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    for d in [args.out_root, args.ray_log]:
        os.makedirs(d, exist_ok=True)
        if not os.access(d, os.W_OK):
            raise PermissionError(f"No write permission for: {d}")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({np.nan: None})


def main():
    parser = argparse.ArgumentParser(description="Normalize video FPS using FFmpeg + Ray")
    parser.add_argument('--is_local', action='store_true', help="Run in local mode (no Ray)")
    parser.add_argument('--fps_required', type=float, default=FPS_DEFAULT)
    parser.add_argument('--fps_tol', type=float, default=FPS_TOL_DEFAULT)
    parser.add_argument('--ray_log', required=True)
    parser.add_argument('--csv_file', required=True, help="Path to input CSV file")
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--no_copy_if_same', action='store_true', help="Force re-encode even if FPS matches")
    parser.add_argument('--src_root', required=True, help="Root directory of source videos")
    parser.add_argument('--num_workers', type=int, default=64, help="Number of Ray workers")

    args = parser.parse_args()
    validate_args(args)

    args_dict = {
        'out_root': args.out_root,
        'src_root': args.src_root,
        'fps_required': args.fps_required,
        'fps_tol': args.fps_tol,
        'overwrite': args.overwrite,
        'no_copy_if_same': args.no_copy_if_same,
    }

    if args.is_local:
        with open(args.csv_file, "r", encoding="utf-8-sig") as f:
            samples = list(csv.DictReader(f))
        worker = Worker(args_dict)
        for item in samples:
            out = worker(item)
            if out['proc_status'] == 1:
                print(out['proc_path_fps25'])
            else:
                print(f"ERROR: {out['proc_error']} | src: {_get_local_path(item)}")
    else:
        print("Reading CSV...")
        df = pd.read_csv(args.csv_file, encoding="utf-8-sig")
        df_clean = clean_dataframe(df)
        print(f"Original: {df.shape}, Cleaned: {df_clean.shape}")

        ds = ray.data.from_pandas(df_clean).repartition(args.num_workers)
        predictions = ds.map(
            lambda item: Worker(args_dict)(item),
            num_cpus=1,
            concurrency=args.num_workers,
        ).map(_csv_sanitize_row)

        output_file = os.path.join(args.ray_log, "fps_processed.csv")
        predictions.write_csv(output_file)
        print(f"✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()