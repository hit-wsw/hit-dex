import os
import json
import cv2

metadata_root = '/media/wsw/SSD1T1/data/metadata'

episode_dirs = sorted([
    d for d in os.listdir(metadata_root)
    if d.startswith('episode_')
], key=lambda x: int(x.split('_')[1]))

for episode in episode_dirs:
    json_path = os.path.join(metadata_root, episode, 'data.json')
    

    # 打开并读取 JSON 数据
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取并打印每个 idx 下的 waist 数据
    for item in data['data']:
        idx = item['idx']
        waist_data = item['states']['waist']['qpos']
        color_jpg_path = os.path.join(metadata_root, episode, item['colors']['color_0'])
        #print(f"{episode} - idx: {idx}, waist: {waist_data}")
        print(idx)
