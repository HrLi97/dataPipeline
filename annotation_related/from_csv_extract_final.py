import pandas as pd

df_filter = pd.read_csv('/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/0919_openhuman_one_human_all_25fps_part_1.csv')      # 只有 file_path
df_full = pd.read_csv('/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/fps/open_part/0919_openhuman_one_human_all_25fps_part_1.csv')            # 完整数据，包含 file_path 和其他列

whitelist = set(df_filter['video_url'])

df_result = df_full[df_full['video_url'].isin(whitelist)]

df_result.to_csv('/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/annotation_data/0919_openhuman_one_human_all_25fps_part_1_final.csv', index=False)