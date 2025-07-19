# from PIL import Image
# import numpy as np

# # 輸入與輸出檔案
# input_image = "ILSVRC2012_val_00000024.JPEG"
# output_bin = "ILSVRC2012_val_00000024_32x32.bin"

# # Resize 成 32x32 並轉為 uint8 陣列
# img = Image.open(input_image).resize((32, 32))
# img_array = np.array(img).astype(np.uint8)

# # 攤平成 1D array（uint8）
# flat = img_array.flatten()

# # 存成 binary 檔案
# flat.tofile(output_bin)

# print(f"Saved uint8 binary to: {output_bin}")

from PIL import Image
import numpy as np
import os

# 輸入圖片檔名列表
input_images = [
    "ILSVRC2012_val_00000001.JPEG",
    "ILSVRC2012_val_00000002.JPEG",
    "ILSVRC2012_val_00000003.JPEG",
    "ILSVRC2012_val_00000004.JPEG",
    "ILSVRC2012_val_00000005.JPEG"
]

# 輸出解析度設定
resolutions = [224, 128, 64, 32]

# 輸出資料夾
output_dir = "imagenet_bin_outputs"
os.makedirs(output_dir, exist_ok=True)

# 處理每張圖片（使用 enumerate 取得 index）
for idx, image_path in enumerate(input_images):
    img = Image.open(image_path).convert("RGB")  # 確保為 RGB

    for res in resolutions:
        resized_img = img.resize((res, res), Image.BILINEAR)
        img_array = np.array(resized_img).astype(np.uint8)
        flat = img_array.flatten()

        output_filename = f"imagenet_img{idx}_{res}x{res}.bin"
        output_path = os.path.join(output_dir, output_filename)
        flat.tofile(output_path)

        print(f"Saved: {output_path}")

