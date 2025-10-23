import csv
import os

def process_csv(input_file, output_file, old_prefix, new_prefix, suffix_to_remove, target_ext=None):
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # 使用csv.reader处理，自动识别带引号的字段（含逗号）
        reader = csv.DictReader(infile)
        # 定义输出CSV的字段
        fieldnames = reader.fieldnames
        
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        # prefix_length = len('/mnt/cfs/shanhai/Datasets/MEAD/video/M003/')
        for row_num, row in enumerate(reader, start=2):  # 行号从2开始（表头为1）
            try:
                src_path = row['vid_path']
                # mark_value = row['mark']
                
                # 替换前缀
                if src_path.startswith(old_prefix):
                    new_path = src_path.replace(old_prefix, new_prefix, 1)
                    # new_path = new_prefix + src_path[prefix_length:]
                else:
                    new_path = src_path
                    print(f"警告：第{row_num}行路径不包含目标前缀，将保持原样")
                
                # 移除后缀
                if new_path.endswith(f"{suffix_to_remove}.mp4"):
                    dir_name, file_name = os.path.split(new_path)
                    new_file_name = file_name.replace(suffix_to_remove, "", 1)
                    if target_ext is not None:
                        new_file_name = os.path.splitext(new_file_name)[0] + target_ext  # 例如 "xxx.mp4" → "xxx.mp3"   
                    new_path = os.path.join(dir_name, new_file_name)
                
                # 写入处理后的行
                writer.writerow({
                    'vid_path': new_path
                })
                
            except Exception as e:
                print(f"处理第{row_num}行时出错：{str(e)}，将跳过该行")
                continue
    
    print(f"处理完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理视频路径CSV文件，替换前缀并移除后缀')
    parser.add_argument('--input', default='/mnt/cfs/shanhai/zhouyang/DataProcess/csv/batch_4_vis_wholebody.csv', help='输入CSV文件路径')
    parser.add_argument('--output', default='/mnt/cfs/shanhai/zhouyang/DataProcess/csv/batch_4_audio_input.csv', help='输出CSV文件路径')
    
    args = parser.parse_args()
    
    old_prefix = "/mnt/cfs/shanhai/zhouyang/DataProcess/vis_wholebody/batch_4"
    new_prefix = "/mnt/cfs/shanhai/zhouyang/DataProcess/dataset/batch_4_cut"  
    suffix_to_remove = "_vis"
    ext = ".mp4"
    process_csv(args.input, args.output, old_prefix, new_prefix, suffix_to_remove, ext)
