#!/usr/bin/env bash
# extract_simInsts.sh

find . -type f -name stats.txt | while read file; do
    sim=$(grep -m1 '^system.cpu.cpi' "$file" | awk '{print $2}')
    dir=$(basename "$(dirname "$file")")
    name=${dir#m5out_}
    echo "$name: $sim"
done

