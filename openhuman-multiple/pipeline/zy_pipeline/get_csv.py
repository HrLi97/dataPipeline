import os
import csv
import argparse

def find_mp4_files(root_dir, extensions='.mp4'):
    """递归查找指定目录下所有的MP4文件，返回绝对路径列表"""
    mp4_paths = []
    # 遍历目录及其子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件扩展名是否为.mp4（不区分大小写）
            if filename.lower().endswith(extensions):
                # 获取文件的绝对路径
                full_path = os.path.abspath(os.path.join(dirpath, filename))
                mp4_paths.append(full_path)
    return mp4_paths

def save_to_csv(file_paths, output_csv):
    """将文件路径列表保存为CSV文件，格式为vid_path列"""
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入CSV
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['vid_path'])
        writer.writeheader()  # 写入表头
        for path in file_paths:
            writer.writerow({'vid_path': path})

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='查找指定文件夹下所有MP4文件并生成CSV列表')
    parser.add_argument('--root', default='/mnt/cfs/shanhai/zhouyang/DataProcess/vis_wholebody/batch_4', help='要查找MP4文件的根目录')
    parser.add_argument('--output', default='./csv/batch_4_vis_wholebody.csv', help='输出的CSV文件路径（例如：./mp4_files.csv）')
    args = parser.parse_args()
    
    # 验证根目录是否存在
    if not os.path.isdir(args.root):
        print(f"错误：指定的根目录不存在 → {args.root}")
        return
 
    extensions='.mp4'
    mp4_files = find_mp4_files(args.root, extensions)
    
    # 处理结果
    if not mp4_files:
        print(f"未找到任何{extensions}文件")
        return
    
    # 保存到CSV
    save_to_csv(mp4_files, args.output)
    print(f"成功找到 {len(mp4_files)} 个{extensions}文件，已保存到 → {args.output}")

if __name__ == "__main__":
    main()
    