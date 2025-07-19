# 读取 filtered_ExecAll_trace.out 中的 tick 值
exec_trace_file = "filtered_ExecAll_trace.out"
MemoryAccess_trace_file = "MemoryAccess_trace.out"
output_file = "filtered_MemoryAccess_trace.out"

# 使用集合来存储所有的 tick 值
ticks = set()

# 读取 filtered_ExecAll_trace.out 文件中的所有 tick
with open(exec_trace_file, "r") as exec_file:
    for line in exec_file:
        tick = line.split(":")[0].strip()
        ticks.add(tick)

# 读取 MemoryAccess_trace.out 并将匹配的行输出到 filtered_MemoryAccess_trace.out
with open(MemoryAccess_trace_file, "r") as memoryaccess_file, open(output_file, "w") as output:
    for line in memoryaccess_file:
        tick = line.split(":")[0].strip()
        if tick in ticks:
            output.write(line)

print(f"Filtered lines have been written to {output_file}")
