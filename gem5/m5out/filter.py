import re

# 输入文件路径
filtered_file = "/RAID2/LAB/css/cssRA01/gem5_old/m5out/filtered_output.txt"
cache_file = "/RAID2/LAB/css/cssRA01/gem5_old/m5out/CacheAll.out"
output_file = "matched_cache_output.txt"

# 读取filtered_output.txt中的所有行，提取每行":"前面的数字
timestamps = set()
with open(filtered_file, 'r') as f:
    for line in f:
        match = re.match(r"(\d+):", line)
        if match:
            timestamps.add(match.group(1))

# 在CacheAll.out中查找包含任何一个数字的行，将匹配行写入output_file
with open(cache_file, 'r') as cache, open(output_file, 'w') as out:
    for line in cache:
        # 检查行开头的数字是否在timestamps中
        line_timestamp = line.split(":")[0]
        if line_timestamp in timestamps:
            out.write(line)

print("Matched lines have been saved to matched_cache_output.txt")
