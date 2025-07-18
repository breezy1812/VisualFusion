import os
import re

src_dir = '/circ330/HomoLabels320240/Version3'
dst_dir = '/circ330/HomoLabels480360/Version3'

src_files = os.listdir(src_dir)
dst_files = os.listdir(dst_dir)

# 建立目標檔案的時間戳字典
# key: HH-MM-SS, value: 完整檔名
pattern_dst = re.compile(r'IR_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.json')
dst_map = {}
for f in dst_files:
    m = pattern_dst.match(f)
    if m:
        time_str = m.group(2)  # HH-MM-SS
        dst_map[time_str] = f

# 320240的檔案格式: IR_HH-MM-SS.json
pattern_src = re.compile(r'IR_(\d{2}-\d{2}-\d{2})\.json')

for f in src_files:
    m = pattern_src.match(f)
    if m:
        time_str = m.group(1)
        if time_str in dst_map:
            src_path = os.path.join(src_dir, f)
            new_name = dst_map[time_str]
            new_path = os.path.join(src_dir, new_name)
            print(f'Renaming {src_path} -> {new_path}')
            os.rename(src_path, new_path)
        else:
            print(f'No match for {f} in 480360 folder')
