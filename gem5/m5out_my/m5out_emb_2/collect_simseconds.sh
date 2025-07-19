#!/usr/bin/env bash
# collect_simseconds.sh
# Scan all datasets and patch directories under m5out_emb_2 and extract simSeconds from each pim/stats.txt

set -euo pipefail

# Base directory (assumes you run this script from /RAID2/LAB/css/cssRA01/gem5/m5out_emb_2)
BASE_DIR="$(pwd)"

# List of datasets to scan
DATASETS=( "CIFAR" "ImageNet" "Flower" "SVHN" )

# Possible patch directories
PATCHES=( "patch_8" "patch_16" "patch_32" "patch_64" )

# Output file
OUTPUT="./simseconds_summary.txt"
echo "# dataset  patch  simSeconds" > "$OUTPUT"

for ds in "${DATASETS[@]}"; do
    for patch in "${PATCHES[@]}"; do
        stats_path="$BASE_DIR/$ds/$patch/pim_2/stats.txt"
        if [ -f "$stats_path" ]; then
            # Find the first line starting with simSeconds and take the second field
            simsec=$(grep -m1 "^system.mem_ctrls.bwTotal::total" "$stats_path" | awk '{print $2}')
            echo "$ds  $patch  $simsec" | tee -a "$OUTPUT"
        fi
    done
done

echo "=== Done: Results written to $OUTPUT ==="

