from PIL import Image
import numpy as np

# 輸入與輸出檔案
input_image = "ILSVRC2012_val_00000024.JPEG"
output_bin = "ILSVRC2012_val_00000024.bin"

# Resize 成 224x224 並轉為 uint8 陣列
img = Image.open(input_image).resize((224, 224))
img_array = np.array(img).astype(np.uint8)

# 攤平成 1D array（uint8）
flat = img_array.flatten()

# 存成 binary 檔案
flat.tofile(output_bin)

print(f"Saved uint8 binary to: {output_bin}")
