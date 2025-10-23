# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import librosa
from tqdm.auto import tqdm
import sys
import os
import torch
import numpy as np
import soundfile as sf
import ray
import csv
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils import demix, get_model_from_config
from clearvoice.clearvoice import ClearVoice
import warnings
warnings.filterwarnings("ignore")


import subprocess
import numpy as np
import os
from tempfile import NamedTemporaryFile

def load_video_audio(video_path, target_sr=44100, mono=False):
    """
    使用 ffmpeg 从视频文件中提取音频，并转换为 numpy 数组
    
    参数:
        video_path: 视频文件路径（支持 MP4、MKV 等含音频流的格式）
        target_sr: 目标采样率（默认 44100 Hz）
        mono: 是否转换为单声道（True 为单声道，False 保留原声道数）
    返回:
        audio: 音频数据 numpy 数组，形状为 (声道数, 采样点数) 或 (采样点数,)（单声道）
        sr: 实际采样率（与 target_sr 一致）
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 构建 ffmpeg 命令：提取音频并转换为 PCM 格式
    # -vn: 禁用视频流
    # -acodec pcm_s16le: 音频编码为 16 位 PCM（无损）
    # -ar: 采样率
    # -ac: 声道数（1 为单声道，0 保留原声道）
    cmd = [
        "ffmpeg",
        "-hide_banner",  # 隐藏版权信息
        "-loglevel", "error",  # 只输出错误信息
        "-i", video_path,  # 输入视频文件
        "-vn",  # 不处理视频
        "-acodec", "pcm_s16le",  # 音频编码为 PCM 16位
        "-ar", str(target_sr),  # 采样率
        "-ac", "1" if mono else "0",  # 声道数（0 表示保留原声道）
        "-f", "s16le",  # 输出格式为 16位小端 PCM
        "-"  # 输出到 stdout
    ]
    
    # 执行 ffmpeg 命令并读取输出
    try:
        # print(video_path)
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 提取音频失败: {e.stderr.decode('utf-8')}")
    
    # 将 PCM 数据转换为 numpy 数组（16位整数 -> 浮点数）
    audio_data = np.frombuffer(result.stdout, dtype=np.int16)
    if len(audio_data) == 0:
        raise ValueError(f"无法从视频中提取音频: {video_path}")
    
    # 计算声道数和采样点数
    # 原视频声道数需要通过 ffmpeg probe 获取（如果 mono=False）
    if not mono:
        # 调用 ffmpeg probe 获取音频流信息
        probe_cmd = [
            "ffprobe",
            "-hide_banner",
            "-loglevel", "error",
            "-show_entries", "stream=channels",
            "-of", "csv=p=0",
            video_path
        ]
        try:
            channels = int(subprocess.check_output(probe_cmd).decode("utf-8").strip())
        except Exception as e:
            raise RuntimeError(f"获取声道数失败: {str(e)}")
        # 重塑为 (声道数, 采样点数)
        audio = audio_data.reshape(-1, channels).T  # 转置为 (channels, samples)
    else:
        # 单声道直接返回 1D 数组
        audio = audio_data.astype(np.float32) / 32767.0  # 归一化到 [-1, 1]
        channels = 1
    
    # 归一化到 [-1, 1]（16位 PCM 最大值为 32767）
    if channels > 1:
        audio = audio.astype(np.float32) / 32767.0
    
    return audio, target_sr

import os
import subprocess

def extract_audio_from_mp4_to_wav(mp4_path):
    """
    使用ffmpeg从MP4文件中提取音频并保存为同目录的WAV文件
    
    参数:
        mp4_path: MP4文件的路径
    返回:
        生成的WAV文件路径，如果失败则返回None
    """
    # 生成WAV文件路径（与MP4同目录，同名不同扩展名）
    wav_path = os.path.splitext(mp4_path)[0] + ".WAV"  # 修正：使用wav_path变量名
    
    # 如果WAV已存在，直接返回路径
    if os.path.exists(wav_path):
        return wav_path
    
    # 使用ffmpeg提取音频并保存为WAV
    # -vn: 不处理视频
    # -acodec pcm_s16le: WAV标准编码器（16位PCM）
    # -y: 覆盖已存在的文件
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", mp4_path,
        "-vn",                  # 移除视频流
        "-acodec", "pcm_s16le", # 关键：使用WAV对应的PCM编码器
        "-y",                   # 覆盖现有文件
        wav_path
    ]
    
    try:
        # print(mp4_path)
        subprocess.run(cmd, check=True)
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"提取WAV音频失败: {e.stderr}")
        return None
    
def run_one(path, model, args, config, device, verbose=False):
    instruments = config.training.instruments.copy()
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]
    
    # 提取音频为MP3
    mp3_path = extract_audio_from_mp4_to_wav(path)
    if not mp3_path:
        return None, None
    
    # 使用librosa加载MP3
    mix, sr = librosa.load(mp3_path, sr=44100, mono=False)
    # mix, sr = load_video_audio(path, target_sr=44100, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    mix_orig = mix.copy()
    if config.inference.get('normalize', False):
        mono = mix.mean(0)
        mean = mono.mean()
        std = mono.std()
        mix = (mix - mean) / std

    track_proc_list = [mix.copy()]
    if args.use_tta:
        track_proc_list.extend([mix[::-1].copy(), -1. * mix.copy()])

    full_result = []
    for mix in track_proc_list:
        waveforms = demix(config, model, mix, device, pbar=not args.disable_detailed_pbar, model_type=args.model_type)
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

    if args.extract_instrumental:
        instr = 'vocals' if 'vocals' in instruments else instruments[0]
        instruments.append('instrumental')
        waveforms['instrumental'] = mix_orig - waveforms[instr]

    save_data = {
        'waveforms': waveforms,
        'sr': sr,
        'mix_orig': mix_orig,
        'instruments': instruments,
        'path': path,
        'normalize_params': (mean, std) if config.inference.get('normalize', False) else None
    }
    return save_data

def SE(path, myClearVoice):
    # 增强
    file_name = path.replace('_vocals', '_SE48K')
    output_wav = myClearVoice(input_path=path, online_write=False)
    #output_file = os.path.join(save_folder, f"{file_name}_SE48K.wav")
    myClearVoice.write(output_wav, output_path=file_name)
    return output_wav, file_name



torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description='audio feat extraction')
parser.add_argument('--store_dir', default="/mnt/cfs/shanhai/zhouyang/DataProcess/enhanced_audio/batch_4")
parser.add_argument('--is_local', default=True)
parser.add_argument("--model_type", type=str, default='mel_band_roformer')
parser.add_argument("--extract_instrumental", action='store_true', default=False)
parser.add_argument("--disable_detailed_pbar", action='store_true')
parser.add_argument("--flac_file", action='store_true')
parser.add_argument("--pcm_type", type=str, default='PCM_24')
parser.add_argument("--use_tta", action='store_true')

args = parser.parse_args()

class Worker:
    def __init__(self):
        model_type = 'mel_band_roformer'
        config_path = './audio_model/config_vocals_mel_band_roformer_kj.yaml'
        start_check_point = './audio_model/MelBandRoformer.ckpt'
        device = "cuda"
        
        model, config = get_model_from_config(model_type, config_path)
        state_dict = torch.load(start_check_point, map_location=device)
        model.load_state_dict(state_dict)
        self.model = model.to(device).eval()
        self.config = config
        self.device = device
        self.myClearVoice = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])

    def save_results(self, data, output_dir):
        path = data['path']
        parent_folder = os.path.basename(os.path.dirname(path))
        file_name = os.path.splitext(os.path.basename(path))[0]
        save_folder = os.path.join(output_dir, parent_folder)
        os.makedirs(save_folder, exist_ok=True)

        results = []
        for instr in data['instruments']:
            estimates = data['waveforms'][instr].T
            if data['normalize_params']:
                mean, std = data['normalize_params']
                estimates = estimates * std + mean

            if args.flac_file:
                output_file = os.path.join(save_folder, f"{file_name}_{instr}.flac")
                subtype = args.pcm_type
            else:
                output_file = os.path.join(save_folder, f"{file_name}_{instr}.wav")
                subtype = 'FLOAT'
            
            sf.write(output_file, estimates, data['sr'], subtype=subtype)
            results.append(output_file)
        return results

    def __call__(self, item):
        try:
            # if args.is_local:
            #     vid_path = item['src']
            #     status = item.get('status', 0)
            # else:

            # 关键：确保路径是字符串
            vid_path = item['vid_path'] 
            if isinstance(vid_path, np.ndarray):
                vid_path = vid_path.item()  # 转换numpy数组为Python标量
            if not isinstance(vid_path, str):
                vid_path = str(vid_path)  # 强制转为字符串            
            
            # status = item.get('status', [0])[0]
            # status = int(float(status))
            # if status != 1:
            #     return item

            # Step 1: 运行源分离 

            separation_data = run_one(vid_path, self.model, args, self.config, self.device)
            # Step 2: 保存分离结果
            saved_files = self.save_results(separation_data, args.store_dir)
            
            # Step 3: 对每个分离结果进行语音增强
            for sep_file in saved_files:
                enhanced_audio, output_file = SE(sep_file, self.myClearVoice)
                # sf.write(output_file, enhanced_audio, 48000, subtype='PCM_16')

            item['status'] = [1] #if not args.is_local else 1
        except Exception as e:
            print(f"Error processing {vid_path}: {str(e)}")
            item['status'] = [0] #if not args.is_local else 0
        return item

csv_file_path = "/mnt/cfs/shanhai/zhouyang/DataProcess/csv/batch_4_audio_input.csv"

# if args.is_local:
# samples = list(csv.DictReader(open(csv_file_path, "r", encoding="utf-8-sig")))

# worker = Worker()

# for item in samples:
#     worker(item)
# else:
ray.init(
    # runtime_env={
    #     "env_vars": {
    #         "PYTHONPATH": ":".join([
    #             current_dir,
    #             "/home/ubuntu/MyFiles/projects/audio_feat/Music-Source-Separation-Training"
    #         ])
    #     }
    # }
)
samples = ray.data.read_csv(csv_file_path)

predictions = samples.map_batches(
    Worker,
    num_gpus=1,
    batch_size=1,
    concurrency=8,
)

predictions.write_csv("/mnt/cfs/shanhai/zhouyang/DataProcess/log/batch4_audio_enhanced")
print("\n================ Done ===============\n")
# 在原来提取wav文件的基础上，由于输入是mp4，新增根据ffmpeg提取音频部分并保存的代码