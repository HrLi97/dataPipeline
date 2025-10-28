import pandas as pd
import json

def csv_to_json(csv_file, json_file, page=0, pageSize=6):
    df = pd.read_csv(csv_file)
    
    videos = []
    for idx, row in df.iterrows():
        video = {
            "src": row["file_path"],
            "mark": 1,
            "type": "video/douyin_record"
        }
        videos.append(video)
    
    result = {
        "page": page,
        "pageSize": pageSize,
        "mark": [0, 1],
        "type": "video",
        "videos": videos
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"生成 JSON 文件成功：{json_file}")

if __name__ == "__main__":
    csv_file = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/part-1-moss.csv"
    json_file = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/part-1-moss.jsonl"
    csv_to_json(csv_file, json_file)