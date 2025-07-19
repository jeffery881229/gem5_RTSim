# import scipy.io
# import numpy as np
# from PIL import Image

# # === Step 1: 載入 SVHN Cropped Dataset ===
# svhn = scipy.io.loadmat("train_32x32.mat")

# # X: shape = (32, 32, 3, N), y: shape = (N,)
# X = svhn['X']  # 圖像資料 (32, 32, 3, N)
# y = svhn['y'].flatten()  # label 資料

# # 修正 label '10' 表示 '0'
# y[y == 10] = 0

# # 取第一張圖
# img_np = X[:, :, :, 0]  # Shape: (32, 32, 3)
# label = int(y[0])

# # === Step 2: Resize 並儲存 ===
# sizes = [32, 64, 128, 224]
# img = Image.fromarray(img_np)

# for size in sizes:
#     resized_img = img.resize((size, size), resample=Image.BILINEAR)

#     # 儲存為 PNG 圖片
#     resized_img.save(f"svhn_img0_{size}x{size}.png")

#     # 儲存為 .bin：1 byte label + RRR... + GGG... + BBB...
#     img_array = np.array(resized_img)
#     r_bin = img_array[:, :, 0].flatten()
#     g_bin = img_array[:, :, 1].flatten()
#     b_bin = img_array[:, :, 2].flatten()
#     bin_data = bytes([label]) + bytes(r_bin) + bytes(g_bin) + bytes(b_bin)

#     with open(f"svhn_img0_{size}x{size}.bin", "wb") as f:
#         f.write(bin_data)
#     print(f"Saved: svhn_img0_{size}x{size}.png + .bin")


import scipy.io
import numpy as np
from PIL import Image

# === Step 1: 載入 SVHN Cropped Dataset ===
svhn = scipy.io.loadmat("train_32x32.mat")

# X: shape = (32, 32, 3, N), y: shape = (N,)
X = svhn['X']
y = svhn['y'].flatten()

# 修正 label '10' 表示 '0'
y[y == 10] = 0

# 尺寸設定
sizes = [32, 64, 128, 224]

# 處理前 5 張圖像
for i in range(5):
    img_np = X[:, :, :, i]  # shape: (32, 32, 3)
    label = int(y[i])
    img = Image.fromarray(img_np)

    for size in sizes:
        resized_img = img.resize((size, size), resample=Image.BILINEAR)

        # 儲存為 PNG
        png_name = f"svhn_img{i}_{size}x{size}.png"
        resized_img.save(png_name)

        # 儲存為 BIN：1 byte label + RRR... + GGG... + BBB...
        img_array = np.array(resized_img)
        r_bin = img_array[:, :, 0].flatten()
        g_bin = img_array[:, :, 1].flatten()
        b_bin = img_array[:, :, 2].flatten()
        bin_data = bytes([label]) + bytes(r_bin) + bytes(g_bin) + bytes(b_bin)

        bin_name = f"svhn_img{i}_{size}x{size}.bin"
        with open(bin_name, "wb") as f:
            f.write(bin_data)

        print(f"Saved: {png_name} + {bin_name}")
