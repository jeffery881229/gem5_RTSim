# from PIL import Image
# import numpy as np

# # === Step 1: 載入圖片（以第一張 image_00001.jpg 為例） ===
# img_path = "image_00001.jpg"
# img = Image.open(img_path).convert("RGB")  # 確保為 RGB 模式

# # 模擬 label，這裡你可以根據實際分類標籤來設定
# label = 0  # e.g., 第 1 類

# # === Step 2: Resize 並輸出 ===
# sizes = [32, 64, 128, 224]

# for size in sizes:
#     resized_img = img.resize((size, size), resample=Image.BILINEAR)

#     # 儲存為 .png
#     png_filename = f"flower_img0_{size}x{size}.png"
#     resized_img.save(png_filename)

#     # 儲存為 .bin (格式為 label + R + G + B)
#     img_array = np.array(resized_img)
#     r_bin = img_array[:, :, 0].flatten()
#     g_bin = img_array[:, :, 1].flatten()
#     b_bin = img_array[:, :, 2].flatten()
#     bin_data = bytes([label]) + bytes(r_bin) + bytes(g_bin) + bytes(b_bin)

#     bin_filename = f"flower_img0_{size}x{size}.bin"
#     with open(bin_filename, "wb") as f:
#         f.write(bin_data)

#     print(f"Saved: {png_filename}, {bin_filename}")

from PIL import Image
import numpy as np
import os

# 處理這些圖片（從 image_00002.jpg 開始到 image_00005.jpg）
input_images = [
    "image_00002.jpg",
    "image_00003.jpg",
    "image_00004.jpg",
    "image_00005.jpg"
]

# 輸出解析度設定
resolutions = [224, 128, 64, 32]

# 輸出資料夾（與當前目錄同一層）
output_dir = "."
# 你也可以自訂資料夾：
# output_dir = "./flower_bin_outputs"
# os.makedirs(output_dir, exist_ok=True)

# 從 img1 開始編號（因為 img0 已經有了）
start_idx = 1

# 執行 resize 並儲存
for i, image_path in enumerate(input_images, start=start_idx):
    img = Image.open(image_path).convert("RGB")

    for res in resolutions:
        resized_img = img.resize((res, res), Image.BILINEAR)
        img_array = np.array(resized_img).astype(np.uint8)
        flat = img_array.flatten()

        output_filename = f"flower_img{i}_{res}x{res}.bin"
        output_path = os.path.join(output_dir, output_filename)
        flat.tofile(output_path)

        print(f"Saved: {output_path}")
