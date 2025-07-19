from PIL import Image
import numpy as np

# === Step 1: 讀取 .bin 檔案 ===
with open("cifar10_img0.bin", "rb") as f:
    content = f.read()

label = content[0]
img_data = np.frombuffer(content[1:], dtype=np.uint8)  # 3072 bytes

# === Step 2: 還原成 RGB 影像 ===
r = img_data[0:1024].reshape((32, 32))
g = img_data[1024:2048].reshape((32, 32))
b = img_data[2048:].reshape((32, 32))
img_np = np.stack([r, g, b], axis=2)  # Shape: (32, 32, 3)

# === Step 3: 使用 PIL 轉成 Image 並 resize ===
img = Image.fromarray(img_np)

# Resize 並儲存為圖片
sizes = [64, 128, 224]
for size in sizes:
    resized_img = img.resize((size, size), resample=Image.BILINEAR)
    resized_img.save(f"cifar10_img0_{size}x{size}.png")
    print(f"Saved: cifar10_img0_{size}x{size}.png")

    # Optional: 如果你還想把 resized 的圖片重新轉回 .bin 格式
    img_array = np.array(resized_img)
    r_bin = img_array[:, :, 0].flatten()
    g_bin = img_array[:, :, 1].flatten()
    b_bin = img_array[:, :, 2].flatten()
    new_bin = bytes([label]) + bytes(r_bin) + bytes(g_bin) + bytes(b_bin)

    with open(f"cifar10_img0_{size}x{size}.bin", "wb") as f:
        f.write(new_bin)
        print(f"Saved .bin: cifar10_img0_{size}x{size}.bin")
