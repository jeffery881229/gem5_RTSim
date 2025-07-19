# 開啟輸入檔案和輸出檔案
input_file = "TLB.out"
output_file = "filtered_TLB.out"

# 打開檔案並處理資料
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        # 檢查該行是否包含 '0x35'
        if "0x35" in line:
            # 將該行寫入輸出檔案
            outfile.write(line)

print(f"Filtering completed. Results are written to {output_file}")
