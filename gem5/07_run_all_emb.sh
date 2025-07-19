#!/usr/bin/env bash
# run_experiments.sh
# Automate experiments across datasets and patch‐size settings (no parallels)

set -euo pipefail

# Configure datasets and their base image sizes
DATASETS=( "CIFAR" "ImageNet" "Flower" "SVHN" )
declare -A SIZES
SIZES=( ["SVHN"]=224 ["Flower"]=128 ["CIFAR"]=32 ["ImageNet"]=64 )

# Paths
GEM5_ROOT="$HOME/gem5"
EMBED_SRC_DIR="$GEM5_ROOT/tests/test-progs/embedding_pim/src"
EMBED_SRC="$EMBED_SRC_DIR/embedding_pim.cpp"
EMBED_COMPILE_SCRIPT="$EMBED_SRC_DIR/01_compile.sh"
PAGE_TABLE="$GEM5_ROOT/src/mem/page_table.cc"
SUBARRAY="$GEM5_ROOT/../RT_SIM_IMC/src/SubArray.cpp"
GEM5_BUILD_SCRIPT="$GEM5_ROOT/02_build_gem5.sh"
RUN_BINARY="$GEM5_ROOT/build/X86/gem5.opt"
RUN_CONFIG="configs/deprecated/example/se.py"
RUN_PROGRAM="tests/test-progs/embedding_pim/bin/x86/linux/embedding_pim"

# Base directories for logs and m5out
LOG_BASE="$GEM5_ROOT/log_emb_2"
M5OUT_BASE="$GEM5_ROOT/m5out_emb_2"

# Helper: mask lower 3 hex digits to zero
mask_hex() {
    python3 -c 'import sys; addr=int(sys.argv[1],16); print(hex(addr & ~0xfff))' "$1"
}

for dataset in "${DATASETS[@]}"; do
    size=${SIZES[$dataset]}

    # Determine which patch sizes to run
    if [ "$dataset" == "CIFAR" ]; then
        PATCH_SIZES=(8 16 32)
    else
        PATCH_SIZES=(8 16 32 64)
    fi

    for patch in "${PATCH_SIZES[@]}"; do
        echo "=== Dataset=${dataset}, imageSize=${size}, patchSize=${patch} ==="

        # 1) Update embedding_pim.cpp: snprintf(...) → point to correct five‐image filenames
        case "$dataset" in
            "CIFAR")
                DS_DIR="CIFAR-10"
                PREFIX="cifar"
                ;;
            "SVHN")
                DS_DIR="SVHN"
                PREFIX="svhn"
                ;;
            "Flower")
                DS_DIR="Oxford_102_Flower"
                PREFIX="flower"
                ;;
            "ImageNet")
                DS_DIR="ImageNet_ILSVRC-2012"
                PREFIX="imagenet"
                ;;
        esac

        # Replace the std::snprintf(...) line in embedding_pim.cpp
        sed -i -E "\|std::snprintf\(filename, sizeof\(filename\),|c\
        std::snprintf(filename, sizeof(filename), \
            \"/RAID2/LAB/css/cssRA01/gem5/data_set/${DS_DIR}/${PREFIX}_img%d_${size}x${size}.bin\", idx);\
        " "$EMBED_SRC"

        # 1b) Update imageH, imageW, and patchSize in embedding_pim.cpp
        sed -i -E "s/(static const int imageH *= *)[0-9]+;/\1${size};/" "$EMBED_SRC"
        sed -i -E "s/(static const int imageW *= *)[0-9]+;/\1${size};/" "$EMBED_SRC"
        sed -i -E "s/(static const int patchSize *= *)[0-9]+;/\1${patch};/" "$EMBED_SRC"

        # 2) Compile embedding code
        echo "[Info] Compiling embedding code..."
        pushd "$EMBED_SRC_DIR" > /dev/null
        bash "$(basename "$EMBED_COMPILE_SCRIPT")"
        popd > /dev/null

        # 3) Extract PIM addresses (quick run w/ parallel=1 in SubArray.cpp)
        echo "[Info] Extracting PIM addresses..."
        sed -i -E "s/(static const int parallel *= *)[0-9]+;/\11;/" "$SUBARRAY"
        # bash "$GEM5_BUILD_SCRIPT"

        extract_out="$M5OUT_BASE/$dataset/patch_$patch/extract_old"
        mkdir -p "$extract_out"
        extract_log="$LOG_BASE/$dataset/patch_$patch/extract_${dataset}_${size}_old.log"
        mkdir -p "$(dirname "$extract_log")"

        nohup "$RUN_BINARY" \
            --outdir="$extract_out" \
            $RUN_CONFIG -c $RUN_PROGRAM \
            --mem-type=NVMainMemory --caches --l2cache --l1i_size 32kB --l1d_size 32kB \
            --l2_size 2MB --cpu-type=X86AtomicSimpleCPU --cpu-clock=3.2GHz --sys-clock=400MHz \
            --nvmain-config=../RT_SIM_IMC/Config/SK.config \
            >& "$extract_log" &
        extract_pid=$!
        wait $extract_pid

        # Parse addresses from extract log
        w_addr=$(grep -m1 "W_patch_quant:"   "$extract_log" | awk '{print $2}')
        b_addr=$(grep -m1 "b_patch_quant:"   "$extract_log" | awk '{print $2}')
        cls_addr=$(grep -m1 "CLS_quant:"     "$extract_log" | awk '{print $2}')
        pos_addr=$(grep -m1 "posEmb_quant:"  "$extract_log" | awk '{print $2}')
        w_mask=$(mask_hex "$w_addr")
        b_mask=$(mask_hex "$b_addr")
        cls_mask=$(mask_hex "$cls_addr")
        pos_mask=$(mask_hex "$pos_addr")

        # 4) Update page_table.cc with new addresses and sizes
        sed -i -E "s/(Addr W_patch_quant *= *)0x[0-9a-fA-F]+;/\1${w_mask};/"   "$PAGE_TABLE"
        sed -i -E "s/(Addr b_patch_quant *= *)0x[0-9a-fA-F]+;/\1${b_mask};/"   "$PAGE_TABLE"
        sed -i -E "s/(Addr CLS_quant *= *)0x[0-9a-fA-F]+;/\1${cls_mask};/"     "$PAGE_TABLE"
        sed -i -E "s/(Addr posEmb_quant *= *)0x[0-9a-fA-F]+;/\1${pos_mask};/"  "$PAGE_TABLE"
        sed -i -E "s/(static const int IMAGE_H *= *)[0-9]+;/\1${size};/"        "$PAGE_TABLE"
        sed -i -E "s/(static const int PATCH_SIZE *= *)[0-9]+;/\1${patch};/"    "$PAGE_TABLE"

        # Update SubArray parameters
        sed -i -E "s/(static const int IMAGE_H *= *)[0-9]+;/\1${size};/"        "$SUBARRAY"
        sed -i -E "s/(static const int PATCH_SIZE *= *)[0-9]+;/\1${patch};/"    "$SUBARRAY"

        # 5) Final gem5 simulation run (no parallels)
        echo "[Info] Running simulation for ${dataset}, patch=${patch}..."
        bash "$GEM5_BUILD_SCRIPT"
        sleep 5
        outdir="$M5OUT_BASE/$dataset/patch_$patch/pim_old"
        mkdir -p "$outdir"
        logf="$LOG_BASE/$dataset/patch_$patch/gem5_${dataset}_${size}_old.log"
        mkdir -p "$(dirname "$logf")"

        nohup "$RUN_BINARY" \
            --outdir="$outdir" \
            $RUN_CONFIG -c $RUN_PROGRAM \
            --mem-type=NVMainMemory --caches --l2cache --l1i_size 32kB --l1d_size 32kB \
            --l2_size 2MB --cpu-type=X86O3CPU --cpu-clock=3.2GHz --sys-clock=400MHz \
            --nvmain-config=../RT_SIM_IMC/Config/SK.config \
            >& "$logf" &
        sleep 5
        # wait

        # echo "[Done] ${dataset} patch=${patch}"
    done

    # ===== 在同一 dataset 的所有 patch 都 submit 後，統一等待 =====
    echo "[Info] All patches launched for ${dataset}. Waiting for them to finish..."
    wait
    echo "[Done] Finished all patches for dataset ${dataset}."
    
done
