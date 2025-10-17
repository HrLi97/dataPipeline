#!/usr/bin/env python3
import ast

# —— 请根据实际路径改这两个变量 —— 
in_file  = '/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/2-out.csv'
out_file = '/home/ubuntu/MyFiles/haoran/code/Data_process_talking/process_for_tvshow/data/batch-1/2-out-mask.csv'

total = kept = 0
with open(in_file,  'r', encoding='utf-8') as fin, \
     open(out_file, 'w', encoding='utf-8') as fout:

    header = fin.readline()
    fout.write(header)

    for line in fin:
        total += 1
        line = line.rstrip('\n')

        i1 = line.find(',')
        i2 = line.rfind(',')
        if i1 < 0 or i2 < 0 or i1 == i2:
            continue

        audio = line[:i1]
        data  = line[i1+1:i2].strip()
        video = line[i2+1:]

        if len(data) >= 2 and data[0] == data[-1] and data[0] in ("'", '"'):
            data = data[1:-1]

        try:
            recs = ast.literal_eval(data)
        except Exception:
            continue

        if not isinstance(recs, list):
            continue

        # 检查 mask 只要有一个非空就 OK
        keep = False
        print(len(recs),"recsrecs")
        for rec in recs:
            # print(rec,"recrec")
            if isinstance(rec, dict):
                m = rec.get('mask', {})
                if m != {}:
                # if isinstance(m, dict) and m:
                    keep = True
                    break

        if keep:
            fout.write(f'{audio},{data},{video}\n')
            kept += 1

print(f'总行数: {total}, 保留行数: {kept}')
