import os
import csv
import argparse

def find_se48k_wav_files(root_dir):
    """
    获取指定目录及子目录下所有后缀为.wav且名称末尾为"_SE48K"的文件路径
    
    参数:
        root_dir: 要搜索的根目录
    返回:
        符合条件的文件路径列表
    """
    target_files = []
    
    # 遍历指定目录及所有子目录
    for root, _, files in os.walk(root_dir):
        for file in files:
            # 检查文件名是否以"_SE48K.wav"结尾
            if file.endswith("_SE48K.wav"):
                # 构建完整路径
                full_path = os.path.abspath(os.path.join(root, file))
                target_files.append(full_path)
    
    return target_files

def save_to_csv(file_paths, output_file):
    """
    将文件路径列表保存到CSV文件，表头为vid_path
    
    参数:
        file_paths: 文件路径列表
        output_file: 输出CSV文件路径
    """
    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        # 创建CSV写入器，指定表头
        writer = csv.DictWriter(csvfile, fieldnames=['vid_path'])
        
        # 写入表头
        writer.writeheader()
        
        # 写入所有文件路径
        for path in file_paths:
            writer.writerow({'vid_path': path})

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='查找指定目录下所有后缀为.wav且名称末尾为"_SE48K"的文件，并保存到CSV')
    parser.add_argument('--dir', default='/mnt/cfs/shanhai/zhouyang/DataProcess/enhanced_audio/batch_4', help='要搜索的根目录路径')
    parser.add_argument('--output', default='./csv/batch_4_se48k_files.csv', help='输出CSV文件路径（例如：se48k_files.csv）')
    
    args = parser.parse_args()
    
    # 验证目录是否存在
    if not os.path.isdir(args.dir):
        print(f"错误：目录 '{args.dir}' 不存在或不是有效目录")
        exit(1)
    
    # 获取符合条件的文件路径
    se48k_files = find_se48k_wav_files(args.dir)
    
    # 保存到CSV
    save_to_csv(se48k_files, args.output)
    
    print(f"成功找到 {len(se48k_files)} 个符合条件的文件")
    print(f"文件路径已保存到：{os.path.abspath(args.output)}")
