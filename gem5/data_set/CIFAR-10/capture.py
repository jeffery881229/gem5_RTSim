# import pickle

# # 載入 CIFAR-10 的第一個 data_batch
# with open("data_batch_1", "rb") as f:
#     batch = pickle.load(f, encoding='bytes')

# # 取出第 0 張圖片的資料與標籤
# img_flat = batch[b'data'][0]       # 長度為 3072 的 uint8 array
# label = batch[b'labels'][0]        # 0~9 的類別數字

# # 將 label + image 合併成一個 bytes
# bin_data = bytes([label]) + bytes(img_flat)

# # 儲存成 .bin 檔案
# with open("cifar10_img0.bin", "wb") as f:
#     f.write(bin_data)

# print("finish: cifar10_img0.bin")

import pickle
import numpy as np
from PIL import Image
import os

# 解壓後的資料夾位置
dataset_path = '/RAID2/LAB/css/cssRA01/gem5/data_set/CIFAR-10'
output_dir = '/RAID2/LAB/css/cssRA01/gem5/data_set/CIFAR-10/dataset2'
os.makedirs(output_dir, exist_ok=True)

# Resize 尺寸列表
resolutions = [224, 128, 64, 32]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 讀取 data_batch_1
batch = unpickle(os.path.join(dataset_path, 'data_batch_1'))

# 取得圖片資料與標籤
data = batch[b'data']
labels = batch[b'labels']

# 處理前五張圖片
for i in range(5):
    img_flat = data[i]
    r = img_flat[0:1024].reshape(32, 32)
    g = img_flat[1024:2048].reshape(32, 32)
    b = img_flat[2048:].reshape(32, 32)
    img = np.stack([r, g, b], axis=2)  # shape: (32, 32, 3)

    pil_img = Image.fromarray(img)

    for res in resolutions:
        resized_img = pil_img.resize((res, res), Image.BILINEAR)  # resize to H x W
        img_np = np.array(resized_img, dtype=np.uint8)

        # 儲存為 .bin 檔案
        filename = f"cifar_img{i}_{res}x{res}.bin"
        filepath = os.path.join(output_dir, filename)
        img_np.tofile(filepath)

print("finish!")
