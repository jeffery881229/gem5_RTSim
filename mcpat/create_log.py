import os
import subprocess

# 設定目錄
input_dir = "/RAID2/LAB/css/cssRA01/gem5/emb_data/xml/"  # XML 文件所在目錄
output_dir = "/RAID2/LAB/css/cssRA01/gem5/emb_data/power/"  # LOG 文件的輸出目錄
mcpat_executable = "./mcpat"  # McPAT 執行檔的路徑

# 遍歷目錄中的所有 .xml 文件
for file_name in os.listdir(input_dir):
    if file_name.endswith(".xml"):
        input_path = os.path.join(input_dir, file_name)
        output_file_name = file_name.replace(".xml", ".log")
        output_path = os.path.join(output_dir, output_file_name)

        # 構建 McPAT 指令
        command = [
            mcpat_executable,
            "-infile", input_path,
            "-print_level", "5"
        ]

        # 執行 McPAT 並將輸出重定向到 .log 文件
        try:
            print(f"Processing {file_name} -> {output_file_name}")
            with open(output_path, "w") as output_file:
                subprocess.run(command, stdout=output_file, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {file_name}: {e}")
