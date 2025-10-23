import argparse
import os
import sys
import os.path as path
import json
import mmcv
import jsonlines
import cv2
import ffmpeg
sys.path.append('/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/openhuman-multiple/utils')
from local_rtmlib import Wholebody, draw_skeleton, PoseTracker
import numpy as np
import ray
import csv
import pandas as pd
import torch

parser = argparse.ArgumentParser(description='分离GPU推理与CPU可视化的姿态估计流水线')
parser.add_argument("--save_path", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/zy_lmks/openhunman-1/part2/",help="可视化结果保存路径")
parser.add_argument("--keypoint_path", default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/zy_lmks/openhunman-1/part2/",help="关键点数据保存路径")
parser.add_argument('--model_arch', default='wholebody', help="模型架构名称")
parser.add_argument('--csv_path', default="/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/fps/open_part/0919_openhuman_one_human_all_25fps_part_2.csv",help="包含视频路径的CSV文件")
opt = parser.parse_args()

def read_vid_paths(csv_file):
    vid_paths = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)  # 使用DictReader可以按列名访问
        for row in reader:
            vid_paths.append(row['video_path_fps25'])  # 提取"vid_path"列的值
    return vid_paths


def pose_to_bbox(keypoints: np.ndarray, expansion: float = 1.25) -> np.ndarray:
    """从关键点获取边界框"""
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    bbox = np.array([x.min(), y.min(), x.max(), y.max()])
    center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2
    bbox = np.concatenate([
        center - (center - bbox[:2]) * expansion,
        center + (bbox[2:] - center) * expansion
    ])
    return bbox


def get_video_dimensions(video_path):
    """获取视频尺寸信息"""
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    if len(video_streams) > 0:
        width = video_streams[0]['width']
        height = video_streams[0]['height']
        return width, height
    else:
        raise ValueError("No video stream found in the file")


def video_inference(input_video, pose_model, opt):
    """仅执行GPU推理，保存关键点数据，不做可视化"""
    # 提前初始化video_name，避免未定义错误
    video_name = path.basename(input_video).split('.')[0] if input_video else "unknown_video"
    print('======== video_name: ', video_name)
    try:
        pose_model.reset()
        vid_reader = mmcv.VideoReader(input_video)
        fps = vid_reader.fps
        height, width = vid_reader[0].shape[:2]

        parent_dir = input_video.split('.mp4')[0].split('/')[-2]
        keypoint_save_path = os.path.join(opt.keypoint_path, parent_dir)
        os.makedirs(keypoint_save_path, exist_ok=True)
        out_jsonl_path = path.join(keypoint_save_path, f'{video_name}_wholebody.jsonl')

        # 跳过已处理的视频
        if os.path.exists(out_jsonl_path):
            print(f"关键点文件 {video_name} 已存在，跳过推理...")
            return True, keypoint_save_path, video_name

        out_dict_list = []
        is_valid = True
        
        for i in range(len(vid_reader)):
            frame = vid_reader[i]
            keypoints, scores = pose_model(frame)   

            if keypoints.shape[0] > 1:  # 多个人物，标记为无效
                is_valid = False
                break

            # 保存关键点数据
            out_dict = dict(
                video_name=video_name, 
                H=height, W=width, 
                frame_idx=i, 
                keypoint=keypoints.tolist(), 
                score=scores.tolist(),
                fps=fps
            )
            out_dict_list.append(out_dict)

        if is_valid:
            with jsonlines.open(out_jsonl_path, mode="w") as f:
                f.write_all(out_dict_list)
            print(f"视频 {video_name} 推理完成，已保存关键点")
        return is_valid, keypoint_save_path, video_name
    except Exception as e:
        print(f"处理视频 {video_name} 时出错: {e}")
        return False, "", video_name


def batch_visualization(keypoint_data_list):
    """批量处理可视化，仅用CPU，并返回原始批次数据"""
    # 转换批次数据格式：从 {key: [vals...]} 转换为元素列表 [{key: val}, ...]
    num_elements = len(keypoint_data_list["vid_path"])
    elements = [
        {
            "vid_path": keypoint_data_list["vid_path"][i],
            "is_valid": keypoint_data_list["is_valid"][i],
            "keypoint_path": keypoint_data_list["keypoint_path"][i],
            "video_name": keypoint_data_list["video_name"][i]
        }
        for i in range(num_elements)
    ]
    
    for data in elements:
        if not data["is_valid"]:
            continue
            
        # 加载关键点数据
        kp_file = os.path.join(data["keypoint_path"], f'{data["video_name"]}_wholebody.jsonl')
        if not os.path.exists(kp_file):
            continue
        
        # 读取视频
        vid_path = data["vid_path"]
        try:
            vid_reader = mmcv.VideoReader(vid_path)
        except Exception as e:
            print(f"无法读取视频 {vid_path}: {e}")
            continue
        
        # 读取关键点
        try:
            with jsonlines.open(kp_file, mode="r") as f:
                kp_list = list(f)
            if not kp_list:
                continue
                
            fps = kp_list[0]["fps"]
            height, width = kp_list[0]["H"], kp_list[0]["W"]
        except Exception as e:
            print(f"处理关键点文件 {kp_file} 错误: {e}")
            continue
        
        # 准备可视化输出
        vis_save_path = os.path.join(opt.save_path, data["keypoint_path"].split('/')[-1])
        os.makedirs(vis_save_path, exist_ok=True)
        save_mp4 = path.join(vis_save_path, f'{data["video_name"]}_vis.mp4')
        if os.path.exists(save_mp4):
            print(f"可视化文件 {save_mp4} 已存在，跳过...")
        else:   
            # 视频编码
            writer = (ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                            s='{}x{}'.format(width, height), r=fps)
                    .output(save_mp4, vcodec='libx264', pix_fmt='yuv420p',
                            video_bitrate="1M", r=fps, loglevel='quiet')
                    .overwrite_output()
                    .run_async(pipe_stdin=True))
            
            # 绘制骨架并写入视频
            try:
                for i, frame in enumerate(vid_reader):
                    if i >= len(kp_list):
                        break
                    kp = np.array(kp_list[i]["keypoint"])
                    scores = np.array(kp_list[i]["score"])
                    img_show = draw_skeleton(frame.copy(), kp, scores, openpose_skeleton=True, kpt_thr=3)
                    writer.stdin.write(np.ascontiguousarray(img_show[:, :, ::-1]).tobytes())
                
                writer.stdin.close()
                writer.wait()
                print(f"可视化完成: {save_mp4}")
            except Exception as e:
                print(f"可视化过程错误 {save_mp4}: {e}")
                try:
                    writer.stdin.close()
                except:
                    pass
    
    # 关键修复：返回原始批次数据（保持流水线数据流通）
    return keypoint_data_list


def load_model():
    """加载姿态估计模型"""
    device = 'cuda'
    backend = 'onnxruntime'
    openpose_skeleton = True
    
    wholebody = PoseTracker(
        Wholebody,
        det_frequency=7,
        to_openpose=openpose_skeleton,
        mode='performance',
        backend=backend,
        tracking=False,
        device=device)

    return wholebody


class InferencePredictor:
    """推理预测器，仅负责GPU推理部分"""
    def __init__(self):
        self.wholebody = load_model()
        
    def __call__(self, batch):
        """处理批次数据，保持与单元素处理逻辑一致"""
        # 当batch_size=1时，批次字典的值都是单元素列表
        vid_path = batch['video_path_fps25']#[0]  # 提取单个视频路径
        
        # 确保路径是字符串（与原始逻辑完全一致）
        if isinstance(vid_path, np.ndarray):
            vid_path = vid_path.item()
        if not isinstance(vid_path, str):
            vid_path = str(vid_path)

        try:
            # 假设video_inference返回(is_valid, kp_path, vid_name)
            is_valid, kp_path, vid_name = video_inference(vid_path, self.wholebody, opt)
            # 保持返回格式为批次字典（值为列表）
            return {
                "vid_path": [vid_path],
                "is_valid": [int(is_valid)],
                "keypoint_path": [kp_path],
                "video_name": [vid_name]
            }
        except Exception as e:
            print(f"!!!!!!!!!!!! Error process {vid_path}")
            print('Reason:', e)
            return {
                "vid_path": [vid_path],
                "is_valid": [0],
                "keypoint_path": [""],
                "video_name": [""]
            }


if __name__ == "__main__":
    # 流式处理：GPU推理 -> 过滤无效结果 -> CPU可视化（流水线执行）
    print("===== 开始流式处理（边推理边可视化） =====")

    # ray.init()
    ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": ":".join([
                        "/mnt/cfs/shanhai/lihaoran/Data_process/dataPipeline/openhuman-multiple/utils",
                    ]),
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "DEEPFACE_HOME": "/home/ubuntu/MyFiles/haoran/cpk/",
                }
            }
        )
    
    samples = ray.data.read_csv(opt.csv_path)
    print("samples.count",samples.count())

    # 重分区：按视频数量分块（每个块对应1个视频，避免批次内数据干扰）
    # 若视频数量极大（如1万+），可设为 samples.count()//2 减少分区数，降低调度开销
    samples = samples.repartition(samples.count() // 2)

    # '''先调试两部分本身有没有问题'''
    # csv_files = opt.csv_path
    # csv_file_path_list = read_vid_paths(csv_files) 
    # gpu_predict = InferencePredictor()
    # for video_path in csv_file_path_list:
    #     out = gpu_predict({"vid_path": video_path})

    predictions = samples.map_batches(
        InferencePredictor,
        num_gpus = 1,
        batch_size=1,
        concurrency=3,
    )

    predictions = predictions.map_batches(
        batch_visualization,
        num_cpus=1,
        batch_size=1,
        concurrency=20,
    )
    write_csv_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/zy_lmks/ray_log/openhuman-1/"   
    predictions.write_csv(write_csv_path, min_rows_per_file=1)

    print("\n===== 全部流程完成 =====")
    ray.shutdown()
