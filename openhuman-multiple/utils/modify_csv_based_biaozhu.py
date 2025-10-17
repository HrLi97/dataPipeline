import pandas as pd

# 通过标注得到的csv去帅选原来csv的内容，并且去重

biaozhu_csv = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/csv_utils/biaozhu_related/tv_show/batch-1/4_22_tvshow-batch-1.csv"
yuan_csv    = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/out_4_22_final_2.csv"
output_csv  = "/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/out_4_22_final_2-filtered_dedup.csv"

df_biao = pd.read_csv(biaozhu_csv)
df_yuan = pd.read_csv(yuan_csv)

# 2. 提取所有有效的 filepath
valid_paths = set(df_biao["file_path"].dropna())

df_filtered = df_yuan[df_yuan["visualize_tracks"].isin(valid_paths)].reset_index(drop=True)

df_deduped = df_filtered.drop_duplicates()

# （可选）打印统计
print(f"过滤后行数：{len(df_filtered)}，去重后行数：{len(df_deduped)}，共删除重复行：{len(df_filtered) - len(df_deduped)}")

# # 5. 保存结果
df_deduped.to_csv(output_csv, index=False)
print(f"已保存过滤并去重后的 CSV：{output_csv}")
