
import csv
import os
import argparse
import ray
import numpy as np
import subprocess

def process(vid_url, output_folder):
    # input_ext = os.path.splitext(vid_url)[1].lower()
    if not isinstance(vid_url, (str, bytes, os.PathLike)):
        raise TypeError(f"process函数需要路径字符串，实际收到: {type(vid_url)}")
    vid_name = os.path.splitext(os.path.basename(vid_url))[0] + '.mp4'
    # up_name_1 = os.path.dirname(vid_url).split('/')[-2]
    # up_name_2 = os.path.dirname(vid_url).split('/')[-1]
    # os.makedirs(os.path.join(output_folder, up_name_1, up_name_2), exist_ok=True)
    # output_file = os.path.join(output_folder, up_name_1, up_name_2, vid_name)
    output_file = os.path.join(output_folder, vid_name)
    if os.path.exists(output_file):
        print(f"文件 {output_file} 已处理，跳过...")
        return 

    command = [
        'ffmpeg',
        '-loglevel', 'error',  # 只输出错误信息，减少干扰
        '-y',                  # 覆盖现有文件
        '-i', vid_url,         # 输入文件
        '-vf', 'fps=25',       # 固定25帧
        '-c:v', 'libx264',     # 视频编码为x264
        '-crf', '10',          # 视频质量
        '-preset', 'veryfast', # 编码速度
        '-c:a', 'aac',         # 将opus音频转码为aac（更通用）
        '-b:a', '128k',        # 音频比特率
        '-strict', '-2',       # 允许实验性功能
        output_file
    ]
    # os.system(command)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(output_file)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg执行失败: {result.stderr}")


parser = argparse.ArgumentParser()
parser.add_argument('--store_dir', default="/mnt/cfs/shanhai/zhouyang/DataProcess/data/batch_4_down/")
parser.add_argument('--is_local', default=False)

args = parser.parse_args()

class Worker:
    def __init__(self):
        pass

    def __call__(self, item):
        if args.is_local:
            vid_path = item['vid_path']
        else:
            vid_path = item['vid_path']
        if isinstance(vid_path, (list, np.ndarray)):
            vid_path = vid_path[0]
    
        try:
            process(vid_path, args.store_dir)

            item['status'] = [1]
        except Exception as e:
            print(f"!!!!!!!!! Error process {vid_path}")
            print('Reason:', e)
            item['status'] = [0]
        return item


csv_file_path = "/mnt/cfs/shanhai/zhouyang/DataProcess/csv/batch_4_left.csv"

# if args.is_local:
# samples = list(csv.DictReader(open(csv_file_path, "r", encoding="utf-8-sig")))
# worker = Worker()
# for item in samples:
#     worker(item)

# else:
samples = ray.data.read_csv(csv_file_path)
predictions = samples.map_batches(
    Worker,
    num_cpus=1,
    batch_size=1,
    concurrency=60,
)

predictions.write_csv("/mnt/cfs/shanhai/zhouyang/DataProcess/log/batch_4_down")   
