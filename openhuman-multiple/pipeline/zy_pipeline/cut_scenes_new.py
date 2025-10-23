#!/usr/bin/env python3
import subprocess
import sys 
sys.path.append("/home/ubuntu/MyFiles/haoran/code/Data_process_talking/") 
sys.path.append("/home/ubuntu/MyFiles/haoran/code/Data_process_talking/utils") 
import argparse
import csv
import os
import ray
import json
# 请根据实际情况导入以下模块
from my_scenedetect.frame_timecode import FrameTimecode
from my_scenedetect.video_splitter import split_video_ffmpeg
# from detect_frames import detect_scenes, detect_scenes_based_time
from detect_frames import detect_scenes, detect_scenes_based_time, hybrid_detect_scenes
import logging
from minio import Minio
import cv2

# DEFAULT_FFMPEG_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 10 -movflags +faststart -c:a aac"
# DEFAULT_FFMPEG_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 8 -c:a aac"
DEFAULT_FFMPEG_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -vf fps=25 -c:v libx264 -preset veryfast -crf 8 -c:a aac"  # 固定25帧
# DEFAULT_FFMPEG_ARGS = "-map 0:v:0 -map 0:a? -map 0:s? -c:v libx264 -preset veryfast -crf 10 -movflags +faststart -c:a aac"

def split_by_max(scene, max_duration):
    start, end = scene
    dur = end - start
    if dur <= max_duration:
        return [(start, end)]
    # 否则二分
    mid = (start + end) / 2
    left  = split_by_max((start, mid), max_duration)
    right = split_by_max((mid, end), max_duration)
    return left + right

def split_scene_slide(scene, max_duration, min_duration):
    start, end = scene
    dur = end - start
    segments = []
    cur = start

    # 用 fixed-window 拆出若干 max_duration 的整段
    while cur + max_duration < end:
        segments.append((cur, cur + max_duration))
        cur += max_duration

    # 把剩余的最后一段当 residual
    residual = (cur, end)
    res_dur = end - cur

    if res_dur >= min_duration or not segments:
        # 如果余数足够长，或者根本没拆出任何整段，就直接保留
        segments.append(residual)
    else:
        prev_start, _ = segments[-1]
        merged = (prev_start, end)
        if merged[1] - merged[0] > max_duration:
            segments[-1:-1] = split_by_max(merged, max_duration)
            segments.pop(-1)
        else:
            segments[-1] = merged

    return segments

def process_interval(interval, video_path, video_name, saved_path, fps):
    start_time, end_time = interval
    sub_scene_duration = end_time - start_time

    if sub_scene_duration < opt.min_duration or sub_scene_duration > opt.max_duration:
        # logging.error(f"Invalid duration ({sub_scene_duration}s) for interval {interval} with FPS ({fps}). Skipping...")
        return

    # 构造输出文件路径
    output_file_name = f"{video_name}_{int(start_time)}-{int(end_time)}.mp4"
    output_file_path = os.path.join(saved_path, output_file_name)

    if os.path.exists(output_file_path):
        return

    try:
        start_timecode = FrameTimecode(start_time, fps=int(fps))
        end_timecode = FrameTimecode(end_time, fps=int(fps))
    except Exception as e:
        logging.error(f"Error converting timecodes for interval {interval}: {e}. Skipping...")
        return

    try:
        split_video_ffmpeg(
            video_path,
            [(start_timecode, end_timecode)],
            output_file_template=output_file_path,
            arg_override=DEFAULT_FFMPEG_ARGS,
            show_progress=False,
        )
    except Exception as e:
        logging.error(f"Failed to split video {video_path} for interval {interval}: {e}")

def get_stream_times(path, threshold=0.1):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,start_time,duration",
        "-of", "json", path
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout)
    print(info,"infoinfoinfoinfoinfo")
    # 分离视频和音频流
    vid = [s for s in info["streams"] if s.get("codec_type") == "video"]
    aud = [s for s in info["streams"] if s.get("codec_type") == "audio"]

    if not vid or not aud:
        return False, info
    
    v0 = vid[0]
    a0 = aud[0]
    print(v0,"v0v0v0v0")
    print(a0,"a0a0a0a0")
    v_start = float(v0.get("start_time", 0.000))
    a_start = float(a0.get("start_time", 0.000))
    v_dur   = float(v0.get("duration", 0.000))
    a_dur   = float(a0.get("duration", 0.000))
    
    # 判断起始时间与时长是否都在阈值以内
    start_ok = abs(v_start - a_start) <= threshold
    dur_ok   = abs(v_dur - a_dur  ) <= threshold
    return (start_ok and dur_ok), {
        "video_start": v_start, "audio_start": a_start,
        "video_duration": v_dur, "audio_duration": a_dur
    }

from urllib.parse import urlparse

def repair_streams(in_path, times, threshold=0.1):
    v_start = times["video_start"]
    a_start = times["audio_start"]
    v_dur = times["video_duration"]
    a_dur = times["audio_duration"]
    print(v_dur,"v_durv_durv_dur")
    print(a_dur,"a_dura_dura_dura_dur")

    parsed = urlparse(in_path)
    filename = os.path.basename(parsed.path)           # e.g. "video.mp4"
    name, ext  = os.path.splitext(filename)   

    base_path = opt.output_fixed_mp4_dir
    os.makedirs(base_path, exist_ok=True)

    out_filename = f"{name}_aligned{ext}"
    out_path = os.path.join(base_path, out_filename)

    # 1) 起始时间修正
    delta = a_start - v_start
    if abs(delta) > threshold:
        if delta > 0:
            # audio starts later → delay video
            #  ffmpeg -itsoffset <delta> -i in.mp4 -i in.mp4 -map 0:v -map 1:a -c copy -shortest out.mp4
            cmd = [
                "ffmpeg",
                "-itsoffset", str(delta),
                "-i", in_path,
                "-i", in_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c", "copy",
                "-shortest",
                "-y",
                out_path
            ]
        else:
            # audio starts earlier → delay audio
            off = -delta
            cmd = [
                "ffmpeg",
                "-i", in_path,
                "-itsoffset", str(off),
                "-i", in_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c", "copy",
                "-shortest",
                "-y",
                out_path
            ]
        subprocess.run(cmd, check=True)
        return out_path

    # 2) 时长修正：如果一个流比另一个长很多，就裁剪到短的那条
    if abs(v_dur - a_dur) > threshold:
        min_dur = min(v_dur, a_dur)
        min_dur_str = f"{min_dur:.5f}"
        print(min_dur_str,"min_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strmin_dur_strv")
        # 这里从新编码
        cmd = [
            "ffmpeg",
            "-i", in_path,
            "-t", min_dur_str,
            "-c:v", "libx264", 
            "-preset", "veryfast",
            "-crf", "10",
            "-c:a", "aac",
            "-b:a", "128k",
            "-y",
            out_path
        ]
        subprocess.run(cmd, check=True)
        return out_path

    return in_path

def check_fix_videoPath(video_path):
    
    aligned, times = get_stream_times(video_path, threshold=0.1)
    if not aligned:
        video_path = repair_streams(video_path, times, threshold=0.1)
        print(f"{video_path} 音视频不同步，详情：{times}，已跳过拆条")
    return video_path

def cut_video(video_path, opt):
    
    video_path = check_fix_videoPath(video_path)

    video_name = os.path.basename(video_path).split('.')[0]
    parent_1 = os.path.basename(os.path.dirname(video_path))
    parent_2 = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    saved_path = os.path.join(opt.output_dir, video_name)
    os.makedirs(saved_path, exist_ok=True)  # 确保输出目录存在

    # 计算视频的基本信息，定义开始时间和end time
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    cap.release()
    
    print(fps,"fpsfpsfpsfpsfpsfpsfpsfpsfps")


    # detect_scenes 返回的是帧区间，此处 fps 为每秒帧数
    # 能不能加入光流 和 多重的判断来定义拆条
    scenes_list, fps = detect_scenes(video_path, 30)
    print(f"视频 {video_path} 检测到的场景帧区间：{scenes_list}")  # 新增打印
    # scenes_list, fps = hybrid_detect_scenes(video_path,psi_threshold=27)

    # 对每个场景，将帧数转换为秒数
    for scene in scenes_list:
        start_frame, end_frame = scene
        scene_start = start_frame / fps
        scene_end = end_frame / fps


        # 之后再对调整后的时间段进行分割
        sub_scenes = split_scene_slide([scene_start, scene_end], opt.max_duration, opt.min_duration)
        if not sub_scenes:
            continue

        if not isinstance(fps, (int, float)) or fps <= 0:
            raise ValueError(f"Invalid FPS value: {fps}")

        for sub_scene in sub_scenes:
            intervals = []
            if (isinstance(sub_scene, (list, tuple)) and len(sub_scene) == 2 and 
                all(isinstance(t, (int, float)) for t in sub_scene)):
                intervals.append(sub_scene)
            # 如果是区间列表
            elif isinstance(sub_scene, (list, tuple)):
                for item in sub_scene:
                    intervals.append(item)
                if not intervals:
                    continue
            else:
                continue

            for interval in intervals:
                process_interval(interval, video_path, video_name, saved_path, fps)


parser = argparse.ArgumentParser(description='scene change detection and transitions detection')
parser.add_argument('--input_csv_path', default='/home/ubuntu/MyFiles/zhouyang/csv/batch_2_diff.csv')
parser.add_argument('--output_dir', default='/home/ubuntu/MyFiles/zhouyang/datasets/batch_2')
parser.add_argument('--max_duration', default='10', type=float)
parser.add_argument('--min_duration', default='3', type=float)
parser.add_argument('--output_fixed_mp4_dir', default='/home/ubuntu/MyFiles/zhouyang/fixed_mp4')
parser.add_argument('--log_path', default='/home/ubuntu/MyFiles/zhouyang/log/log_nice.log', help="日志地址")
parser.add_argument('--is_local', default=True, type=float)
parser.add_argument("--ray_log_dir", type=str, default="/home/ubuntu/MyFiles/zhouyang/log/batch_2/")
opt = parser.parse_args()

class Worker:
    def __init__(self):
        pass

    def __call__(self, item):

        if opt.is_local:
            video_path = item['vid_path']
        else:
            video_path = item['vid_path'][0]
        try:
            cut_video(video_path,opt)
            item['status'] = [1]
        except Exception as e:
            print(f"!!!!!!!!! Error process {item}")
            print('Reason:', e)
            item['status'] = [0]
        return item

if __name__ == '__main__':
    from accelerate import PartialState
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index

    os.makedirs(os.path.dirname(opt.log_path), exist_ok=True,mode=0o777)
    logging.basicConfig(
            filename=opt.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    workers = Worker()
    if opt.is_local:
        samples = list(csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig")))
        with distributed_state.split_between_processes(samples, apply_padding=True) as sample:
            # print(sample,"samplessasamplemplessamples")
            for item in samples:
                print(item,"itemitemitem")
                workers(item)

    else:
        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ":".join([
                        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/utils",
                        "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/"
                    ]),
                    "HF_ENDPOINT": "https://hf-mirror.com",
                }
            }
        )
        os.makedirs(os.path.dirname(opt.ray_log_dir), exist_ok=True,mode=0o777)
        samples = ray.data.read_csv(opt.input_csv_path)
        predictions = samples.map_batches(
            Worker,
            num_cpus=2,
            batch_size=1,      
            concurrency=40,
        )
        predictions.write_csv(opt.ray_log_dir)

