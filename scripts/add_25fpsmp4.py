import pandas as pd
path = "/mnt/cfs/shanhai/lihaoran/Data_process/pipeline_for_openhuman/tmp/fps/25fps_csv/OpenHumanVid_part_003.sample10.csv"
df = pd.read_csv(path)
df["path"] = df["path"].astype(str) + "__fps25.mp4"
df.to_csv(path, index=False)