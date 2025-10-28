import os
import json

def fix_jsonl_file(file_path):
    fixed_lines = []
    changed = False
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                # 如果读出来的是 dict, 说明缺少外层引号
                if isinstance(obj, dict):
                    fixed_lines.append(json.dumps(line))  # 给最外层加一层字符串
                    changed = True
                else:
                    fixed_lines.append(line)
            except json.JSONDecodeError:
                # 如果 decode 失败，说明它已经是正确的字符串（外层有引号）
                fixed_lines.append(line)

    if changed:
        print(f"修复: {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for l in fixed_lines:
                f.write(l + "\n")

def process_folder(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for fname in files:
            if fname.endswith(".jsonl"):
                fix_jsonl_file(os.path.join(root, fname))

if __name__ == "__main__":
    folder = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/zy_lmks/openhunman-1"
    process_folder(folder)
