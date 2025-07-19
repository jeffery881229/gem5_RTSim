#!/bin/bash

base_dir="~/gem5/m5out_ImageNet_emb/patch_64"
base_dir=$(eval echo $base_dir)

results=()

for dir in "$base_dir"/m5out_ImageNet_64_p*/; do
    p_label=$(basename "$dir")                  # 例如：m5out_SVHN_224_p1
    stats_file="$dir/stats.txt"

    if [[ -f "$stats_file" ]]; then
        sim_sec=$(grep "^simSeconds" "$stats_file" | awk '{print $2}')
        p_num=$(echo "$p_label" | grep -oP 'p\K[0-9]+')
        results+=("$p_num $p_label : $sim_sec")
    fi
done

# 排序並輸出（以 p 編號做數字排序）
printf "%s\n" "${results[@]}" | sort -n | cut -d' ' -f2- 
