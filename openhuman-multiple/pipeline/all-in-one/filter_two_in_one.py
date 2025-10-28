# -- coding: utf-8 --
import json
import logging
import math
import os
import re
import subprocess
import time
import cv2
import mmcv
import ray
import sys
from scipy import signal
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import csv
import librosa
import shutil
from accelerate import PartialState
import ffmpeg
import soundfile as sf
from decord import VideoReader, cpu
import shlex
from collections import defaultdict
from moviepy.editor import VideoFileClip, AudioFileClip


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/mmpose-main")
sys.path.insert(
    0, "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/Music-Source-Separation-Training"
)
sys.path.insert(0, "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/syncnet_python-master")

# sys.path.insert(0, "/home/ubuntu/Grounded-SAM-2")
# sys.path.insert(0, "/home/ubuntu/MyFiles/haoran/code/")
# sys.path.insert(0, "/home/ubuntu/MyFiles/haoran/code/Data_process_talking")
# sys.path.insert(0, "/home/ubuntu/Grounded-SAM-2/grounding_dino")

sys.path.append("/mnt/cfs/shanhai/lihaoran/Data_process/a6000")
sys.path.append(
    "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/"
)
sys.path.append(
    "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/grounding_dino/"
)
sys.path.append(
    "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/sam2_opt/sam2"
)


from utils.detect_body import *
from process_for_tvshow.pipeline.utils import *

# from ProcessVideo_ray import *
from scipy.io import wavfile
from SyncNetInstance import *
from detectors import S3FD
from clearvoice.clearvoice import ClearVoice
import easyocr
from utils_separation import demix, get_model_from_config
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T
import torchvision.transforms as T1

GROUNDING_DINO_CONFIG = "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = (
    "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/ckps/groundingdino_swint_ogc.pth"
)
BOX_THRESHOLD_HAND = 0.35
BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.4
PERSON_CONFIDENCE = 0.5
import pycocotools.mask as mask_util
import pandas as pd
from rtmlib import Wholebody, draw_skeleton, PoseTracker
import argparse
from typing import Tuple
from pyannote.audio import Pipeline

AUTH_TOKEN = os.getenv("HF_TOKEN")
local_files_only = True

# from old_pipeline.speaker_recognition_0_ray import *
try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def load_model_mm(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()


def merge_time_speaker_intervals(time_speaker, tolerance=3.0):
    if not time_speaker:
        return []

    time_speaker.sort(key=lambda x: x[0])
    merged = [time_speaker[0]]

    for current in time_speaker[1:]:
        last = merged[-1]
        if current[0] - last[1] <= tolerance and current[2] == last[2]:
            merged[-1] = [last[0], max(last[1], current[1]), last[2]]
        else:
            merged.append(current)
    return merged


if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_image(image_rgb) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image_rgb)
    image_transformed, _ = transform(image_source, None)
    return image_transformed


def load_model_pose(face_only: bool = True):
    device = "cuda"
    backend = "onnxruntime"
    wholebody = Wholebody(
        to_openpose=True, mode="performance", backend=backend, device=device
    )

    # face 在 wholebody 输出里，从索引 25 开始，连续 70 点（取前 68）
    face_start = 25
    face_n = 68

    if face_only:

        def pose_face(img):
            # 调用原模型
            kpts_out, scores_out = wholebody(img)
            kpts = kpts_out[0] if isinstance(kpts_out, (list, tuple)) else kpts_out
            scores = (
                scores_out[0] if isinstance(scores_out, (list, tuple)) else scores_out
            )

            if kpts.ndim == 2:
                kpts = kpts[None, ...]
            if scores.ndim == 1:
                scores = scores[None, ...]

            kp68 = kpts[0, face_start : face_start + face_n, :2]  # (68,2)
            sc68 = scores[0, face_start : face_start + face_n]  # (68,)

            return kp68, sc68

        return pose_face

    else:
        # 返回全身 + 脸 + 手 + 脚
        return wholebody


def crop_video_fan_based_time(track, frames, frame_idxs):
    cropped_frames = []

    for frame_idx, bbox in track:
        if frame_idx not in frame_idxs:
            continue
        if frame_idx < 0 or frame_idx >= len(frames):
            continue

        frame = frames[frame_idx]
        cs = 0.40
        bs = max((bbox[3] - bbox[1]), (bbox[2] - bbox[0])) / 2
        bsi = int(bs * (1 + 2 * cs))

        padded = np.pad(
            frame, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=110
        )

        my = (bbox[1] + bbox[3]) / 2 + bsi
        mx = (bbox[0] + bbox[2]) / 2 + bsi
        face = padded[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]

        if face.size == 0:
            print(f"Warning: Empty crop for frame {frame_idx}, skipping.")
            continue
        try:
            face_resized = cv2.resize(face, (224, 224))
        except Exception as e:
            print(f"Error resizing frame {frame_idx}: {e}")
            continue

        cropped_frames.append(face_resized)

    return np.array(cropped_frames)


def extract_audio_by_track(audio_data, frame_idx, fps=25, sample_rate=16000):
    snippets = []
    for fidx in frame_idx:
        t0 = fidx / fps
        t1 = (fidx + 1) / fps
        s0 = int(round(t0 * sample_rate))
        s1 = int(round(t1 * sample_rate))
        snippets.append(audio_data[s0:s1])
    if snippets:
        return np.concatenate(snippets)
    else:
        return np.zeros(0, dtype=audio_data.dtype)


def gen_audio_by_speakerID(audio_data, all_speaker_data):
    speaker_segments = defaultdict(list)
    for entry in all_speaker_data:
        speaker_segments[entry["speaker_id"]].append(
            (entry["start_time"], entry["end_time"])
        )

    # 获取音频参数
    sr = getattr(opt, "audio_sample_rate", 16000)
    total_samples = len(audio_data)
    is_stereo = audio_data.ndim == 2

    speaker_audio_dir = os.path.join(opt.saved_vid_root, "speaker_full_audio")
    os.makedirs(speaker_audio_dir, exist_ok=True)
    speaker_audio_paths = {}

    for spk_id, segments in speaker_segments.items():
        # 初始化全零音频
        if is_stereo:
            full_audio = np.zeros_like(audio_data)
        else:
            full_audio = np.zeros(total_samples, dtype=audio_data.dtype)

        for start_t, end_t in segments:
            s = int(round(start_t * sr))
            e = int(round(end_t * sr))
            s = max(0, s)
            e = min(total_samples, e)
            if s < e:
                if is_stereo:
                    full_audio[s:e, :] = audio_data[s:e, :]
                else:
                    full_audio[s:e] = audio_data[s:e]

        out_path = os.path.join(
            speaker_audio_dir, f"{opt.reference}_speaker{spk_id}_syncnet_valid.wav"
        )
        sf.write(out_path, full_audio, sr)
        speaker_audio_paths[spk_id] = out_path
        print(
            f"Saved SyncNet-validated full-length audio for speaker {spk_id} to: {out_path}"
        )

    return speaker_audio_paths


def extract_track_audio(audio_data, track, frame_rate, sample_rate=16000):
    audiostart = track[0]  # 获取音频段的起始时间（以秒为单位）
    audioend = track[-1]  # 获取音频段的结束时间（以秒为单位）

    # 转换为样本索引
    start_sample = int(audiostart * sample_rate)
    end_sample = int(audioend * sample_rate)

    # 从音频数据中提取相应的音频片段
    selected_audio = audio_data[start_sample:end_sample]

    return selected_audio


def read_video_with_decord(video_path, use_gpu=False):
    try:
        device = cpu(0)
        vr = VideoReader(video_path, ctx=device)
        first_frame = vr[0]
        if hasattr(first_frame, "asnumpy"):
            height, width = first_frame.asnumpy().shape[:2]
        elif hasattr(first_frame, "numpy"):
            height, width = first_frame.numpy().shape[:2]
        else:
            raise TypeError(f"Unsupported frame data type: {type(first_frame)}")
        total_frame = len(vr)
        return vr, total_frame, height, width

    except RuntimeError as e:
        raise RuntimeError(f"Error reading video file: {video_path}\n{str(e)}")


def calculate_IOU(box1, box2):

    if isinstance(box1[0], list):
        x_coords = [point[0] for point in box1]
        y_coords = [point[1] for point in box1]
        box1 = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    if isinstance(box2, str):
        box2 = json.loads(box2)

    box1 = np.array(box1).flatten()
    box2 = np.array(box2).flatten()

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def run_one(path, model, opt, config, device, verbose=False):
    instruments = config.training.instruments.copy()
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    mix, sr = librosa.load(path, sr=44100, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_orig = mix.copy()
    if config.inference.get("normalize", False):
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

    track_proc_list = [mix.copy()]
    if opt.use_tta:
        track_proc_list.extend([mix[::-1].copy(), -1.0 * mix.copy()])

    full_result = []
    for mix in track_proc_list:
        waveforms = demix(
            config,
            model,
            mix,
            device,
            pbar=not opt.disable_detailed_pbar,
            model_type="mel_band_roformer",
        )
        full_result.append(waveforms)

    waveforms = full_result[0]
    for i in range(1, len(full_result)):
        d = full_result[i]
        for el in d:
            if i == 2:
                waveforms[el] += -1.0 * d[el]
            elif i == 1:
                waveforms[el] += d[el][::-1].copy()
            else:
                waveforms[el] += d[el]
    for el in waveforms:
        waveforms[el] /= len(full_result)

    # if opt.extract_instrumental:
    #     instr = 'vocals' if 'vocals' in instruments else instruments[0]
    #     instruments.append('instrumental')
    #     waveforms['instrumental'] = mix_orig - waveforms[instr]

    save_data = {
        "waveforms": waveforms,
        "sr": sr,
        "mix_orig": mix_orig,
        "instruments": instruments,
        "vid_path": path,
        "normalize_params": (
            (mean, std) if config.inference.get("normalize", False) else None
        ),
    }
    return save_data


def inference_faces(DET, frames):
    dets = []
    for fidx, frame in enumerate(frames):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = DET.detect_faces(frame_rgb, conf_th=0.5, scales=[0.2])
        dets.append(
            [
                {"frame": fidx, "bbox": bbox[:-1].tolist(), "conf": bbox[-1]}
                for bbox in bboxes
            ]
        )
    return dets


def SE(path, myClearVoice):
    file_name = path.replace("_vocals", "_SE48K")
    output_wav = myClearVoice(input_path=path, online_write=False)
    myClearVoice.write(output_wav, output_path=file_name)
    return output_wav, file_name


def process_audio_files(wav_paths, pipeline):

    time_speaker = []
    wav_path = wav_paths

    print(wav_paths, "wav_pathswav_paths")

    try:
        annotations = pipeline(wav_path)
        for turn, _, speaker in annotations.itertracks(yield_label=True):
            speaker_idx = int(speaker[-1])
            curr_ = [float(f"{turn.start:.1f}"), float(f"{turn.end:.1f}"), speaker_idx]
            time_speaker.append(curr_)
    except Exception as e:
        logging.error(f"Error processing {wav_path}: {e}")
        time_speaker = []

    # 融合间隔较短的最长说话人数据
    # time_speaker = merge_time_speaker_intervals(time_speaker, tolerance=0.5)

    # 丢弃时长不足0.3s的片段
    time_speaker = [seg for seg in time_speaker if seg[1] - seg[0] >= 0.2]

    return {
        "time_speaker": time_speaker,
    }


def process_item_person_2(
    vid_path,
    vr,
    saved_vid_root,
    grounding_model,
    image_predictor,
    video_predictor,
    tag,
    BOX_THRESHOLD_1,
):
    # 初始化视频预测状态
    inference_state = video_predictor.init_state(video_path=vid_path)
    TEXT_PROMPT = f"{tag}."
    bbox_dict = {}
    video_segments = {}
    mask_dict = {}
    OBJECTS = []
    fps = vr.get_avg_fps()
    out_width, out_height = vr[0].shape[1], vr[0].shape[0]

    person_num = 0
    valid_two_person = True
    for frame_idx in range(0, len(vr), 5):
        frame_np = vr[frame_idx].numpy()
        boxes, confs, labels = predict(
            model=grounding_model,
            image=load_image(frame_np),
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD,
        )
        person_num = max(person_num, len(boxes))

        if person_num != 2:
            valid_two_person = False
            break

    print(person_num, "person_numperson_numperson_num")

    if not valid_two_person:
        return bbox_dict, mask_dict, video_segments, OBJECTS

    for frame_idx in range(0, len(vr), 5):
        image_source = vr[frame_idx].numpy()
        image = load_image(image_source)

        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD_1,
            text_threshold=TEXT_THRESHOLD,
        )

        if len(boxes) >= person_num:
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(
                boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
            ).numpy()
            confidences = confidences.numpy().tolist()
            OBJECTS = labels

            for object_id, (label, box) in enumerate(
                zip(OBJECTS, input_boxes), start=1
            ):
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    box=box,
                )

            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            break

    # last_name = os.path.basename(vid_path)
    # visualize(vid_path, video_segments, fps, out_width, out_height, f"/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/img/test_{last_name}.mp4", vr, OBJECTS)

    for frame_idx, segments in video_segments.items():
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )

        if frame_idx not in bbox_dict:
            bbox_dict[frame_idx] = {}

        # 第一部分：使用 video_segments 中的 object_ids 初始化 bbox
        for i, obj_id in enumerate(object_ids):
            bbox = detections.xyxy[i]
            if obj_id not in bbox_dict[frame_idx]:
                bbox_dict[frame_idx][obj_id] = []
            bbox_dict[frame_idx][obj_id].append(bbox.tolist())

        # 第二部分：确保对于内层循环中的每个 object_id，都已经初始化对应的键
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            if object_id not in bbox_dict[frame_idx]:
                bbox_dict[frame_idx][object_id] = []  # 如果不存在则初始化
            bbox_dict[frame_idx][object_id].append(box.tolist())

        for i, obj_id in enumerate(object_ids):
            mask = detections.mask[i]
            rle = mask_util.encode(np.asfortranarray(mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            # mask_dict.setdefault(obj_id, []).append(rle)
            mask_dict.setdefault(obj_id, {})[frame_idx] = rle

    return bbox_dict, mask_dict, video_segments, OBJECTS


def save_audio_segment(audio_data, all_speaker_data):

    audio_dir = os.path.join(opt.saved_vid_root, "audio_segments")
    os.makedirs(audio_dir, exist_ok=True)

    for idx, entry in enumerate(all_speaker_data):
        start_t = entry["start_time"]
        end_t = entry["end_time"]
        speaker_id = entry["speaker_id"]
        pid = entry["pid"]

        sr = 16000
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)

        # 边界保护
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_data), end_sample)

        segment_audio = audio_data[start_sample:end_sample]

        # 保存为 WAV 文件
        audio_out_path = os.path.join(
            audio_dir, f"{opt.reference}_speaker{speaker_id}_pid{pid}_seg{idx}.wav"
        )
        sf.write(audio_out_path, segment_audio, sr)
        print(f"Saved audio segment to {audio_out_path}")
        # 可选：将音频路径也写入 best_entry 或 CSV
        entry["audio_segment_path"] = audio_out_path


def visualize(
    vid_path,
    video_segments,
    fps,
    out_width,
    out_height,
    save_vis_file_path,
    vr,
    OBJECTS,
):

    writer_vis = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(out_width, out_height),
            r=fps,
        )
        .output(
            save_vis_file_path,
            vcodec="libx264",
            pix_fmt="yuv444p",
            video_bitrate="10M",
            r=fps,
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

    for frame_idx, segments in video_segments.items():
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)  # Todo
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks,  # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(
            scene=vr[frame_idx].numpy().copy(), detections=detections
        )
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            annotated_frame,
            detections=detections,
            labels=[ID_TO_OBJECTS[i] for i in object_ids],
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        writer_vis.stdin.write(np.ascontiguousarray(annotated_frame).tobytes())
    writer_vis.stdin.close()
    writer_vis.wait()


# # 没有音频轨道,视频流应该是h264
# from moviepy import VideoFileClip, AudioFileClip
# def visualize_tracks_with_audio(video_reader,
#                                 person_bbox_dict,
#                                 all_speaker_data,
#                                 output_path,
#                                 fps,
#                                 source_video_path,kp_radius=3, kp_color=(0,0,255)):
#     spf = {}
#     for rec in all_speaker_data:
#         sid = rec['speaker_id']
#         for f, rle in rec['mask'].items():
#             poses = rec['poses'].get(f, {})
#             spf.setdefault(f, []).append((sid, rle, poses))

#     frame0 = video_reader[0].numpy()
#     h, w, _ = frame0.shape
#     # 'avc1' is the fourcc for H.264 in many builds of OpenCV
#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

#     for idx in range(len(video_reader)):
#         frame = video_reader[idx].numpy()
#         vis   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # draw each speaker's mask + keypoints
#         for sid, rle, poses in spf.get(idx, []):
#             m = mask_util.decode(rle)
#             if m.ndim == 3 and m.shape[2] == 1:
#                 m = m[:,:,0]
#             mask_bin = m.astype(np.uint8)
#             mask_vis = (mask_bin * 255).astype(np.uint8)
#             colored  = np.zeros_like(vis)
#             colored[:,:,1] = mask_vis
#             vis = cv2.addWeighted(vis, 1.0, colored, 0.5, 0)

#             for xg, yg in poses.get('keypoints', []):
#                 cv2.circle(vis, (int(xg), int(yg)), kp_radius, kp_color, -1)
#             if poses.get('keypoints'):
#                 x0, y0 = map(int, poses['keypoints'][0])
#                 cv2.putText(vis, f"S:{sid}", (x0, max(y0-10,10)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#         writer.write(vis)

#     writer.release()

#     video_clip = VideoFileClip(output_path)
#     audio_clip = AudioFileClip(source_video_path)
#     final_clip = video_clip.set_audio(audio_clip)

#     final_path = output_path.replace('.mp4', '_final.mp4')
#     final_clip.write_videofile(
#         final_path,
#         codec='libx264',       # H.264
#         audio_codec='aac',     # AAC for audio
#         temp_audiofile='tmp-audio.m4a',
#         remove_temp=True
#     )
#     return final_path


def visualize_tracks(
    video_reader,
    person_bbox_dict,
    all_speaker_data,
    output_path,
    fps,
    kp_radius=3,
    kp_color=(0, 0, 255),
):
    spf = {}
    for rec in all_speaker_data:
        sid = rec["speaker_id"]
        for f, rle in rec["mask"].items():
            poses = rec["poses"].get(f, {})
            spf.setdefault(f, []).append((sid, rle, poses))

    frame0 = video_reader[0].numpy()
    h, w, _ = frame0.shape
    cmd = (
        f"ffmpeg -y "
        f"-f rawvideo -pixel_format bgr24 -video_size {w}x{h} -framerate {fps} -i - "
        f"-c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p "
        f"{output_path}"
    )
    proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)

    for idx in range(len(video_reader)):
        frame = video_reader[idx].numpy()
        vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for sid, rle, poses in spf.get(idx, []):
            mask_arr = mask_util.decode(rle)
            if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
                mask_arr = mask_arr[:, :, 0]
            mask_bin = mask_arr.astype(np.uint8) * 128
            colored = np.zeros_like(vis)
            colored[:, :, 1] = mask_bin
            vis = cv2.addWeighted(vis, 1.0, colored, 0.3, 0)

            kpts = poses.get("keypoints", [])
            for xg, yg in kpts:
                cv2.circle(vis, (int(xg), int(yg)), kp_radius, kp_color, -1)
            if kpts:
                x0, y0 = map(int, kpts[0])
                cv2.putText(
                    vis,
                    f"S:{sid}",
                    (x0, max(y0 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        proc.stdin.write(vis.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with code {proc.returncode}")


def calculate_mouth_movement(keypoints, prev_mouth_position=None):
    if keypoints is None or len(keypoints) == 0:
        return None, None, None

    mouth_keypoints = keypoints[85:96]
    print(len(keypoints), "keypointskeypointskeypoints")
    print(len(mouth_keypoints), "len(mouth_keypoints)len(mouth_keypoints)")

    if len(mouth_keypoints) < 10:
        return None, None, None

    # 计算上下嘴唇的距离，通常我们可以选择下唇的两个关键点（下唇中间的两个点）和上唇的两个关键点（上唇中间的两个点）
    # 假设使用下唇（87, 88）和上唇（94, 95）的两个点计算
    upper_lip = (mouth_keypoints[2] + mouth_keypoints[3]) / 2  # 上唇的平均点
    lower_lip = (mouth_keypoints[9] + mouth_keypoints[10]) / 2  # 下唇的平均点

    # 计算嘴巴的开合度
    mouth_opening = np.linalg.norm(upper_lip - lower_lip)

    # 计算嘴巴的运动
    mouth_movement = 0
    if prev_mouth_position is not None:
        mouth_movement = np.linalg.norm(
            mouth_keypoints.mean(axis=0) - prev_mouth_position
        )

    return mouth_opening, mouth_movement, mouth_keypoints.mean(axis=0)


def track_for_landmarks(
    tracks,
    vr,
):
    pass


class Worker:
    def __init__(self):
        """
        大体流程：
        1. 先增强，再过滤，再判断
        1. 使用声源分离和增强,拿纯人声的wav和视频结合
        利用process_audio_files来判断有几个声音，判断
        3. 增强 判断有几个声音 遍历生源，遍历

        TODO 超过三个人的片段就直接丢弃了

        2. 使用gounded-sam2检测嘴部mask,得到一个个小视频,使用syncnet分别计算得到参数,置信度过低计算帧之间mask的面积变化
        3. 保存结果,即筛选音画不同步,多人的场景可以,但是只能一个人说话,需要说话人的mask
        """
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device="cuda",
        )
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        model_cfg = "sam2.1_hiera_l.yaml"
        sam2_checkpoint = "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/checkpoints/sam2.1_hiera_large.pt"

        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        model_type = "mel_band_roformer"
        config_path = "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/Music-Source-Separation-Training/ckpts/config_vocals_mel_band_roformer_kj.yaml"
        start_check_point = "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/Music-Source-Separation-Training/ckpts/MelBandRoformer.ckpt"
        device = "cuda"

        model, config = get_model_from_config(model_type, config_path)
        state_dict = torch.load(start_check_point, map_location=device)
        model.load_state_dict(state_dict)
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.myClearVoice = ClearVoice(
            task="speech_enhancement", model_names=["MossFormer2_SE_48K"]
        )

        self.syncnet = SyncNetInstance()
        self.syncnet.loadParameters(opt.initial_syncnet)
        self.detector = S3FD("cuda")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=AUTH_TOKEN,
        )
        self.pipeline.to(torch.device("cuda"))

        self.pose_model = load_model_pose()

    def save_and_enhance_results(self, data, output_dir):
        path = data["vid_path"]
        parent_folder = os.path.basename(os.path.dirname(path))
        file_name = os.path.splitext(os.path.basename(path))[0]
        save_folder = os.path.join(output_dir, parent_folder)
        os.makedirs(save_folder, exist_ok=True)

        final_files = []
        for instr in data["instruments"]:
            estimates = data["waveforms"][instr].T
            if data["normalize_params"]:
                mean, std = data["normalize_params"]
                estimates = estimates * std + mean

            output_file = os.path.join(save_folder, f"{file_name}_{instr}.wav")
            subtype = "FLOAT"
            sf.write(output_file, estimates, data["sr"], subtype=subtype)
            print(f"Saved {instr} file: {output_file}")

            if instr == "vocals":
                enhanced_file = output_file.replace("_vocals", "_SE48K")
                # 调用 ClearVoice 进行增强
                output_wav = self.myClearVoice(
                    input_path=output_file, online_write=False
                )
                self.myClearVoice.write(output_wav, output_path=enhanced_file)
                print(f"Enhanced vocals saved: {enhanced_file}")
                final_files.append(enhanced_file)
            else:
                final_files.append(output_file)

        return final_files

    def save_results(self, data, output_dir, depth=2):
        vid_path = data["vid_path"]
        file_name = os.path.splitext(os.path.basename(vid_path))[0]
        # path - /mnt/cfs/shanhai/lihaoran/data/OpenHumanVid-final/part_003/33/32/3332623f3da02201ffa02a7eba1e643d
        parts = Path(vid_path).parent.parts
        if len(parts) <= 1:
            rel_subdir = ""
        else:
            clean_parts = [p for p in parts if p]
            rel_subdir = os.path.join(*clean_parts[-depth:]) if clean_parts else ""

        save_folder = os.path.join(output_dir, rel_subdir)
        os.makedirs(save_folder, exist_ok=True)

        results = []
        for instr in data["instruments"]:
            estimates = data["waveforms"][instr].T
            if data["normalize_params"]:
                mean, std = data["normalize_params"]
                estimates = estimates * std + mean
            # RGZC-17_104-108_SE48K
            output_file = os.path.join(save_folder, f"{file_name}_SE48K.wav")
            subtype = "FLOAT"

            sf.write(output_file, estimates, data["sr"], subtype=subtype)
            results.append(output_file)

        return results

    def first_step(self, vid_path):
        separation_data = run_one(vid_path, self.model, opt, self.config, self.device)
        # final_files = self.save_and_enhance_results(separation_data, opt.store_dir)
        saved_files = self.save_results(separation_data, opt.store_dir, depth=2)
        print(saved_files, "saved_filessaved_filessaved_files")
        for sep_file in saved_files:
            output_wav, file_name = SE(sep_file, self.myClearVoice)
        return output_wav, file_name

    def detect_mouth_keypoint(self, vidtracks, ii):
        from mmpose.apis import init_model as init_pose_estimator
        from mmpose.utils import adapt_mmdet_pipeline

        detector = init_detector(opt.det_config, opt.det_checkpoint, device="cuda:0")
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)

        pose_config = "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/mmpose-main/configs/face_2d_keypoint/rtmpose/face6/rtmpose-m_8xb256-120e_face6-256x256.py"
        pose_checkpoint = "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/mmpose-main/configs/face_2d_keypoint/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth"

        pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device="cuda:0",
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
        )

        print(len(vidtracks), "vidtracksvidtrackslennnnnnnnnnnn")
        prev_mouth_position = None
        total_mouth_movement = 0
        valid_frames = 0

        movement_time_window = 10
        movement_threshold = 0

        mouth_movements = []
        for frame_idx, frame in enumerate(vidtracks):
            pred_instances = process_one_image(opt, frame, detector, pose_estimator)
            keypoints = pred_instances["keypoints"]

            frame_vis = draw_keypoints(
                frame.copy(), keypoints, pred_instances["keypoint_scores"], kp_thr=0.5
            )
            # mmcv.imwrite(
            #     frame_vis,
            #     f"/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/img/111_frame_{frame_idx}_{ii}_valid.png",
            # )

            mouth_opening, mouth_movement, current_mouth_position = (
                calculate_mouth_movement(keypoints, prev_mouth_position)
            )
            if mouth_opening == None:
                return None
            prev_mouth_position = current_mouth_position

            mouth_movements.append(
                {
                    "frame_idx": frame_idx,
                    "mouth_opening": mouth_opening,
                    "mouth_movement": mouth_movement,
                }
            )

            total_mouth_movement += mouth_movement
            valid_frames += 1

            print(
                f"Frame {frame_idx}: Mouth Opening: {mouth_opening}, Mouth Movement: {mouth_movement}"
            )

            if (
                valid_frames >= movement_time_window
                and total_mouth_movement >= movement_threshold
            ):
                frame_vis = draw_keypoints(
                    frame.copy(),
                    keypoints,
                    pred_instances["keypoint_scores"],
                    kp_thr=0.5,
                )
                mmcv.imwrite(
                    frame_vis,
                    f"/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/img/111_frame_{frame_idx}_{ii}_valid.png",
                )
                print("嘴巴有足够的运动痕迹")
            elif valid_frames >= movement_time_window:
                print("嘴巴的运动量不足，不满足要求")
                return None

        return mouth_movements

    """
    1. 加载视频 → 2. 人脸检测与跟踪（SAM2） → 3. 音频说话人分割 →
    4. 遍历每个说话人片段 → 5. 遍历所有人脸 track →
    6. 裁剪人脸视频 + 提取对应音频 → 7. SyncNet 验证 →
    8. 选置信度最高的 track 作为该说话人 →
    9. 输出 mask/pose/可视化/CSV
    """

    def second_step(
        self,
        video_path,
        audio_path,
        file_name,
        opt,
        grounding_model,
        image_predictor,
        video_predictor,
        pipeline,
        pose_model,
    ):
        opt.reference = os.path.basename(video_path).split(".")[0]
        parentdir1 = os.path.basename(os.path.dirname(video_path))
        print("succussfully load video!!", video_path)
        vr = VideoReader(video_path)
        # frames = vr.get_batch(range(len(vr))).asnumpy()
        valid = False
        # 调用sam2来检测人脸，返回我所有的person - frame - bbox
        person_bbox_dict, bbox_mask, video_segments, OBJECTS = process_item_person_2(
            video_path,
            vr,
            opt.saved_vid_root + "/person",
            grounding_model,
            image_predictor,
            video_predictor,
            "face",
            BOX_THRESHOLD,
        )

        print(person_bbox_dict, "person_bbox_dictperson_bbox_dictperson_bbox_dict")
        print(bbox_mask, "bbox_maskbbox_mask")

        all_person_ids = set()
        for frame_data in person_bbox_dict.values():
            all_person_ids.update(frame_data.keys())

        tracks = {pid: [] for pid in all_person_ids}
        for frame_idx in sorted(person_bbox_dict.keys()):
            frame_data = person_bbox_dict[frame_idx]
            for pid, bboxes in frame_data.items():
                if len(bboxes) > 0:
                    tracks[pid].append((frame_idx, bboxes[0]))

        alltracks = [(pid, tracks[pid]) for pid in sorted(tracks.keys())]

        # 这里需要将音频分段
        audio_path = file_name
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)
        process_audio = process_audio_files(audio_path, pipeline)

        print(process_audio, "process_audioprocess_audioprocess_audioprocess_audio")

        time_speaker = process_audio["time_speaker"]
        all_speaker_data = []
        for start_time, end_time, speaker_id in time_speaker:
            masks_in_range = {}
            # 这里指选择一个
            best_conf = -1.0
            best_entry = None
            start_frame = int(round(start_time * opt.frame_rate))
            end_frame = int(round(end_time * opt.frame_rate))
            # 对每一个片段进行裁剪 分块
            for pid, track in alltracks:
                track_poses = {}
                for fidx, bbox in track:
                    if (
                        fidx < start_frame
                        or fidx > end_frame
                        or fidx < 0
                        or fidx >= len(vr)
                    ):
                        continue
                    frame_full = vr[fidx].numpy()  # RGB
                    x1, y1, x2, y2 = map(int, bbox)
                    face_crop = frame_full[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    keypoints, scores = pose_model(face_crop)
                    # 兼容 (68,2) 或 (1,68,2) 输出
                    kpts = keypoints[0] if keypoints.ndim == 3 else keypoints
                    scs = scores[0] if scores.ndim == 2 else scores
                    abs_kps = []
                    for kp in kpts:
                        xr, yr = kp[0], kp[1]
                        abs_kps.append([x1 + xr, y1 + yr])
                    track_poses[fidx] = {"keypoints": abs_kps, "scores": scs.tolist()}

                frame_idx = [
                    fidx for fidx, _ in track if start_frame <= fidx <= end_frame
                ]
                print(frame_idx, "frame_idxframe_idxframe_idx")
                # 目前来说，这里是对整个人脸的bbox处理，而不是帧对应的，现在我需要让音频帧进行对应！
                # vidtracks = crop_video_fan(opt, track, vr)
                vidtracks = crop_video_fan_based_time(track, vr, frame_idx)
                # print(track_poses,"track_posestrack_posestrack_poses")
                # track_audio = extract_track_audio(audio_data, [start_time, end_time], opt.frame_rate)
                track_audio = extract_audio_by_track(audio_data, frame_idx)
                if len(vidtracks) == 0 or len(track_audio) == 0:
                    continue
                offset, conf, dist = self.syncnet.evaluate(vidtracks, track_audio)
                # 判断syncnet结果
                print(conf, "confcconfcconfconf")
                print(offset, "offsetoffsetoffsetoffset")
                if (
                    opt.min_offset <= offset <= opt.max_offset
                    and conf >= opt.min_confidence
                ):
                    if conf > best_conf:
                        best_conf = conf
                        pid_masks_full = bbox_mask.get(pid, {})
                        for fidx in range(start_frame, end_frame + 1):
                            rle = pid_masks_full.get(fidx)
                            if rle is not None:
                                masks_in_range[fidx] = rle

                        best_entry = {
                            "start_time": start_time,
                            "end_time": end_time,
                            "offset": offset,
                            "conf": conf,
                            "speaker_id": speaker_id,
                            "pid": pid,
                            "mask": masks_in_range,
                            "poses": track_poses,
                        }

            if masks_in_range:
                valid = True
                all_speaker_data.append(best_entry)

        if valid:
            # 保证说话人至少大于25帧
            filtered_speaker_data = [
                entry
                for entry in all_speaker_data
                if len(entry["mask"]) >= 25 and len(entry["poses"]) >= 25
            ]
            speaker_groups = defaultdict(list)
            for entry in filtered_speaker_data:
                speaker_groups[entry["speaker_id"]].append(entry)

            valid_speakers = {}
            for spk_id, entries in speaker_groups.items():
                if entries:
                    valid_speakers[spk_id] = entries[0]

            if len(valid_speakers) < 2:
                return

            all_speaker_data = list(valid_speakers.values())

            # 保存说话人音频片段
            speaker_audio_paths = gen_audio_by_speakerID(
                audio_data=audio_data, all_speaker_data=all_speaker_data
            )

            # 按照speaker_id 分组 all_speaker_data
            speaker_entries = defaultdict(list)
            for entry in all_speaker_data:
                speaker_entries[entry["speaker_id"]].append(entry)
            jdir = os.path.join(opt.saved_vid_root, "jsonl")
            os.makedirs(jdir, exist_ok=True)
            mask_j = os.path.join(jdir, f"{opt.reference}_mask.jsonl")
            pose_j = os.path.join(jdir, f"{opt.reference}_landmarks.jsonl")

            total_frames = len(vr)
            fps = getattr(opt, "frame_rate", vr.get_avg_fps())
            speaker_output_paths = {}

            for spk_id, entries in speaker_entries.items():
                # 聚合该 speaker 的 mask 和 pose
                mask_per_frame = {}
                pose_per_frame = {}

                for rec in entries:
                    for f, r in rec["mask"].items():
                        cnt = r["counts"]
                        if isinstance(cnt, (bytes, bytearray)):
                            cnt = cnt.decode("utf-8")
                        mask_per_frame[f] = {"size": r["size"], "counts": cnt}
                    for f, pts in rec["poses"].items():
                        pose_per_frame[f] = pts

                mask_j = os.path.join(
                    jdir, f"{opt.reference}_mask_speaker_{spk_id}.jsonl"
                )
                pose_j = os.path.join(
                    jdir, f"{opt.reference}_landmarks_speaker_{spk_id}.jsonl"
                )

                with open(mask_j, "w", encoding="utf-8") as mf, open(
                    pose_j, "w", encoding="utf-8"
                ) as pf:
                    for f in range(total_frames):
                        mf.write(
                            f"{f}:"
                            + json.dumps(mask_per_frame.get(f), ensure_ascii=False)
                            + "\n"
                        )
                        pf.write(
                            f"{f}:"
                            + json.dumps(pose_per_frame.get(f), ensure_ascii=False)
                            + "\n"
                        )

                speaker_output_paths[spk_id] = {
                    "mask_jsonl": mask_j,
                    "pose_jsonl": pose_j,
                }

            vis_dir = os.path.join(opt.saved_vid_root, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            vis_noaudio_path = os.path.join(vis_dir, f"{opt.reference}_visnoaudio.mp4")
            fps = getattr(opt, "frame_rate", vr.get_avg_fps())
            visualize_tracks(
                video_reader=vr,
                person_bbox_dict=person_bbox_dict,
                all_speaker_data=all_speaker_data,
                output_path=vis_path,
                fps=fps,
            )

            # 合并原始音频（假设 video_path 包含音频）
            vis_path = os.path.join(vis_dir, f"{opt.reference}_vis.mp4")
            try:
                video_clip = VideoFileClip(vis_noaudio_path)
                audio_clip = AudioFileClip(video_path)
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(
                    vis_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="tmp-audio.m4a",
                    remove_temp=True,
                    logger=None,
                )
                video_clip.close()
                audio_clip.close()
                os.remove(vis_noaudio_path)
            except Exception as e:
                print(f"Failed to add audio to visualization: {e}")
                vis_path = vis_noaudio_path

            print(f"Visualization saved to {vis_path}")

            # 汇总
            with open(opt.output_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(
                        [
                            "audio_path",
                            "speaker_id",
                            "total_frames",
                            "mask_jsonl",
                            "pose_jsonl",
                            "speaker_audio_paths",
                            "video_path",
                            "visualize_tracks",
                        ]
                    )
                writer.writerow(
                    [
                        audio_path,
                        spk_id,
                        total_frames,
                        mask_j,
                        pose_j,
                        speaker_audio_paths,
                        video_path,
                        vis_path,
                    ]
                )
            print(f"Results written to {opt.output_csv_path}")

        else:
            print(
                "$$$$$$$$$$$$$$$$$$$$ No valid speaking segments found, skipping result saving.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            )
            # os.remove(audio_path)
            # failed_dir = os.path.join(
            #     opt.saved_vid_root, "failed_videos_tvshow_batch-1"
            # )
            # os.makedirs(failed_dir, exist_ok=True)
            # dest_path = os.path.join(failed_dir, os.path.basename(video_path))
            # try:
            #     shutil.move(video_path, dest_path)
            #     print(f"Video moved to failed dir: {dest_path}")
            # except Exception as e:
            #     print(f"Failed to move video {video_path} to {failed_dir}: {e}")

    def __call__(self, item):
        if opt.is_local:
            vid_path = item["path"]
        else:
            vid_path = item["path"][0]

        # try:
        output_wav, file_name = self.first_step(vid_path)
        print(output_wav, "output_wavoutput_wavoutput_wav")
        self.second_step(
            vid_path,
            output_wav,
            file_name,
            opt,
            self.grounding_model,
            self.image_predictor,
            self.video_predictor,
            self.pipeline,
            self.pose_model,
        )
        item["status"] = [1]
        # except Exception as e:
        #     print(f"!!!!!!!!! Error process {vid_path}")
        #     print("Reason:", e)
        #     item["status"] = [0]
        return item


parser = argparse.ArgumentParser(description="human segmentation")
parser.add_argument(
    "--input_csv_path",
    default=None,
)
parser.add_argument(
    "--output_csv_path",
    default=None,
)
parser.add_argument(
    "--initial_syncnet",
    type=str,
    default=None,
    help="Path to the SyncNet model.",
)
parser.add_argument(
    "--store_dir",
    default=None,
)
parser.add_argument(
    "--saved_vid_root",
    default=None,
)
parser.add_argument(
    "--log_path",
    default=None,
    help="日志地址",
)
parser.add_argument(
    "--ray_log_dir",
    type=str,
    default=None,
)

parser.add_argument(
    "--det-config",
    default=None,
    help="Config file for detection",
)
parser.add_argument(
    "--det-checkpoint",
    default=None,
    help="Checkpoint file for detection",
)

parser.add_argument("--disable_detailed_pbar", action="store_true")
parser.add_argument("--min_track", type=int, default=1, help="Minimum track length")
parser.add_argument("--flac_file", action="store_true")
parser.add_argument("--crop_scale", type=float, default=0.40, help="Scale bounding box")
parser.add_argument("--pcm_type", type=str, default="PCM_24")
parser.add_argument("--use_tta", action="store_true")
parser.add_argument("--frame_rate", type=int, default=25, help="")
parser.add_argument("--batch_size", type=int, default=1, help="")
parser.add_argument("--vshift", type=int, default=10, help="")
parser.add_argument("--is_local", type=bool, default=False)
parser.add_argument(
    "--num_failed_det",
    type=int,
    default=25,
    help="Number of missed detections allowed before tracking is stopped",
)
parser.add_argument(
    "--min_face_size", type=int, default=100, help="Minimum face size in pixels"
)
parser.add_argument("--max_offset", type=int, default=6, help="Max Video-Audio offset")
parser.add_argument("--min_offset", type=int, default=-6, help="Min Video-Audio offset")
parser.add_argument(
    "--min_confidence", type=int, default=1, help="Minimum confidence value"
)
opt = parser.parse_args()

if __name__ == "__main__":
    os.makedirs(os.path.dirname(opt.log_path), exist_ok=True, mode=0o777)
    logging.basicConfig(
        filename=opt.log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    os.makedirs(os.path.dirname(opt.ray_log_dir), exist_ok=True, mode=0o777)

    if opt.is_local:
        samples = list(
            csv.DictReader(open(opt.input_csv_path, "r", encoding="utf-8-sig"))
        )
        distributed_state = PartialState()
        device_id = distributed_state.local_process_index
        device = f"cuda:{device_id}"
        pred = Worker()
        with distributed_state.split_between_processes(
            samples, apply_padding=True
        ) as sample:
            for item in sample:
                pred(item)

    else:
        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ":".join(
                        [
                            "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/mmpose-main",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/Music-Source-Separation-Training",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/a6000/syncnet_python-master",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt",
                            "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/third_part/Grounded_SAM2_opt/sam2_opt",
                            # "/home/ubuntu/Grounded-SAM-2",
                            # "/home/ubuntu/Grounded-SAM-2/grounding_dino",
                            # "/home/ubuntu/MyFiles/haoran/code",
                            # "/home/ubuntu/MyFiles/haoran/code/Data_process_talking",
                        ]
                    ),
                    "HF_ENDPOINT": "https://hf-mirror.com",
                }
            },
        )
        samples = ray.data.read_csv(opt.input_csv_path)
        predictions = samples.map_batches(
            Worker,
            num_gpus=0.5,
            batch_size=1,
            concurrency=2,
        )
        predictions.write_csv(opt.ray_log_dir)
