from minio import Minio
import csv
import os

# MinIO配置
endpoint = "10.1.200.150"
minio_client = Minio(
    endpoint,
    access_key="TmCrrtYpiMQ1aB1O",  # 替换成实际的ak
    secret_key="ud6tj4EQuOpA3NNFfMNo0E4zCYdyI6Lt",  # 替换为实际的sk
    secure=False
)

bucket_name = "shanhai"
file = "lihaoran/data/annotation/openhuman/part-1/"
# mos/shanhai/ruidi/datasets/OpenHumanVid

objects = minio_client.list_objects(bucket_name, prefix=file, recursive=True)

output_dir = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/part-1_moss.csv"
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

with open(output_dir, "w", newline="") as f:
    writer = csv.writer(f)

    # 如果 CSV 文件为空，则写入表头（包含 file_path 和 flag 两列）
    if os.stat(output_dir).st_size == 0:
        csv_headers = ["file_path"]
        writer.writerow(csv_headers)

    for obj in objects:
        filepath = obj.object_name

        if filepath.rstrip("/") == file:
            continue
        
        if filepath.endswith(".csv"):
            try:
                file_stat = minio_client.stat_object(bucket_name, filepath)
                print(file_stat.size, "file_statfile_statfile_stat")
                if file_stat.size > 10:
                    url = os.path.join(f"http://{endpoint}", bucket_name, filepath)
                    # 写入视频路径及 flag=0
                    writer.writerow([url])
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
