#!/usr/bin/env bash
# run_experiments.sh
# Automate experiments across datasets, patch sizes, and parallel settings
set -euo pipefail

# Configure datasets and sizes
DATASETS=( "CIFAR" "ImageNet" "Flower" "SVHN")
declare -A SIZES
SIZES=( ["SVHN"]=224 ["Flower"]=128 ["CIFAR"]=32 ["ImageNet"]=64 )

# Patch sizes and parallel list
PATCH_SIZES=(8 16 32 64)
PARALLELS=(1)
# PARALLELS=(1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 30 40 50 70 100 196 384)

# Paths
GEM5_ROOT="/RAID2/LAB/css/cssRA01/gem5"
EMBED_SRC_DIR="$GEM5_ROOT/tests/test-progs/embedding_pim/src"
EMBED_SRC="$EMBED_SRC_DIR/embedding_pim.cpp"
EMBED_COMPILE_SCRIPT="$EMBED_SRC_DIR/01_compile.sh"
PAGE_TABLE="$GEM5_ROOT/src/mem/page_table.cc"
SUBARRAY="$GEM5_ROOT/../RT_SIM_IMC/src/SubArray.cpp"
GEM5_BUILD_SCRIPT="$GEM5_ROOT/02_build_gem5.sh"
RUN_BINARY="$GEM5_ROOT/build/X86/gem5.opt"
RUN_CONFIG="configs/deprecated/example/se.py"
RUN_PROGRAM="tests/test-progs/embedding_pim/bin/x86/linux/embedding_pim"

# Helper: mask lower 3 hex digits to zero
mask_hex() {
    python3 -c 'import sys; addr=int(sys.argv[1],16); print(hex(addr & ~0xfff))' "$1"
}

for dataset in "${DATASETS[@]}"; do
    size=${SIZES[$dataset]}
    log_root="${dataset}_log_2"
    mkdir -p "$log_root"

    for patch in "${PATCH_SIZES[@]}"; do
        echo "=== $dataset: image=${size}, patch=${patch} ==="

        # 1) Update embedding_pim.cpp parameters
        # 1a) Update dataset file path
        if [ "$dataset" == "CIFAR" ]; then
            DS_DIR="CIFAR-10"
            DS_FILE="cifar10_img0_${size}x${size}.bin"
        elif [ "$dataset" == "SVHN" ]; then
            DS_DIR="SVHN"
            DS_FILE="svhn_img0_${size}x${size}.bin"
        elif [ "$dataset" == "Flower" ]; then
            DS_DIR="Oxford_102_Flower"
            DS_FILE="flower_img0_${size}x${size}.bin"
        else
            DS_DIR="ImageNet_ILSVRC-2012"
            DS_FILE="ILSVRC2012_val_00000024_${size}x${size}.bin"
        fi
        DATA_PATH="$GEM5_ROOT/data_set/$DS_DIR/$DS_FILE"
                sed -i -E "s|fopen\(\"[^\"]*\.bin\"|fopen(\"${DATA_PATH}\"|" "$EMBED_SRC"

        # 1b) Update embedding parameters
        sed -i -E "s/(static const int imageH *= *)[0-9]+;/\1${size};/" "$EMBED_SRC"
        sed -i -E "s/(static const int imageW *= *)[0-9]+;/\1${size};/" "$EMBED_SRC"
        sed -i -E "s/(static const int patchSize *= *)[0-9]+;/\1${patch};/" "$EMBED_SRC"

        # 2) Compile embedding code (run in source dir)
        echo "Compiling embedding code in $EMBED_SRC_DIR..."
        pushd "$EMBED_SRC_DIR" > /dev/null
        bash "$(basename "$EMBED_COMPILE_SCRIPT")"
        popd > /dev/null

        # 3) Extract PIM addresses (quick run with parallel=1)
        echo "Extracting PIM addresses..."
        sed -i -E "s/(static const int parallel *= *)[0-9]+;/\11;/" "$SUBARRAY"
        bash "$GEM5_BUILD_SCRIPT"
        extract_out="m5out_${dataset}_emb/patch_${patch}/extract"
        mkdir -p "$extract_out"
        extract_log="${log_root}/patch_${patch}/extract_${dataset}_${size}.log"
        mkdir -p "$(dirname "$extract_log")"
        nohup "$RUN_BINARY" \
            --outdir="$extract_out" \
            $RUN_CONFIG -c $RUN_PROGRAM \
            --mem-type=NVMainMemory --caches --l2cache --l1i_size 32kB --l1d_size 32kB \
            --l2_size 2MB --cpu-type=X86AtomicSimpleCPU --cpu-clock=3.2GHz --sys-clock=400MHz \
            --nvmain-config=../RT_SIM_IMC/Config/SK.config \
            >& "$extract_log" &
        wait

        # Parse addresses
        w_addr=$(grep -m1 "W_patch_quant:" "$extract_log" | awk '{print $2}')
        b_addr=$(grep -m1 "b_patch_quant:" "$extract_log" | awk '{print $2}')
        cls_addr=$(grep -m1 "CLS_quant:"   "$extract_log" | awk '{print $2}')
        pos_addr=$(grep -m1 "posEmb_quant:" "$extract_log" | awk '{print $2}')
        w_mask=$(mask_hex "$w_addr")
        b_mask=$(mask_hex "$b_addr")
        cls_mask=$(mask_hex "$cls_addr")
        pos_mask=$(mask_hex "$pos_addr")

        # 4) Update page_table.cc
        sed -i -E "s/(Addr W_patch_quant *= *)0x[0-9a-fA-F]+;/\1${w_mask};/" "$PAGE_TABLE"
        sed -i -E "s/(Addr b_patch_quant *= *)0x[0-9a-fA-F]+;/\1${b_mask};/" "$PAGE_TABLE"
        sed -i -E "s/(Addr CLS_quant *= *)0x[0-9a-fA-F]+;/\1${cls_mask};/" "$PAGE_TABLE"
        sed -i -E "s/(Addr posEmb_quant *= *)0x[0-9a-fA-F]+;/\1${pos_mask};/" "$PAGE_TABLE"
        sed -i -E "s/(static const int IMAGE_H *= *)[0-9]+;/\1${size};/" "$PAGE_TABLE"
        sed -i -E "s/(static const int PATCH_SIZE *= *)[0-9]+;/\1${patch};/" "$PAGE_TABLE"

        # Update SubArray parameters
        sed -i -E "s/(static const int IMAGE_H *= *)[0-9]+;/\1${size};/" "$SUBARRAY"
        sed -i -E "s/(static const int PATCH_SIZE *= *)[0-9]+;/\1${patch};/" "$SUBARRAY"

        # 5) Run simulations for each parallel
        echo "Running parallels for patch=${patch}..."
        pids=()
        for par in "${PARALLELS[@]}"; do
            echo "  parallel=${par}"
            sed -i -E "s/(static const int parallel *= *)[0-9]+;/\1${par};/" "$SUBARRAY"
            bash "$GEM5_BUILD_SCRIPT"
            outdir="m5out_${dataset}_emb/patch_${patch}/m5out_${dataset}_${size}_p${par}"
            mkdir -p "$outdir"
            logf="${log_root}/patch_${patch}/gem5_${dataset}_${size}_p${par}.log"
            mkdir -p "$(dirname "$logf")"
            nohup "$RUN_BINARY" \
                --outdir="$outdir" \
                $RUN_CONFIG -c $RUN_PROGRAM \
                --mem-type=NVMainMemory --caches --l2cache --l1i_size 32kB --l1d_size 32kB \
                --l2_size 2MB --cpu-type=X86O3CPU --cpu-clock=3.2GHz --sys-clock=400MHz \
                --nvmain-config=../RT_SIM_IMC/Config/SK.config \
                >& "$logf" &
            pids+=("$!")
            sleep 1
        done

        echo "Waiting for parallel jobs to finish..."
        for pid in "${pids[@]}"; do wait $pid; done
        echo "Completed patch=${patch}"
    done
done
