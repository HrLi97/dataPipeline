import json
import csv

input_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/0919_openhuman_one_human_all_25fps_part_1.json"
output_path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/0919_openhuman_one_human_all_25fps_part_1.csv"

with open(input_path, "r", encoding="utf-8") as fin, \
     open(output_path, "w", newline="", encoding="utf-8-sig") as fout:

    writer = csv.writer(fout)
    writer.writerow(["file_path"])

    for line in fin:
        record = json.loads(line)
        for video in record.get("videos", []):
            if video.get("mark") == 1:
                writer.writerow([video.get("src")])
