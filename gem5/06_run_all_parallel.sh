#!/usr/bin/env bash
# run_all_parallel.sh
# 使用前請 chmod +x run_all_parallel.sh

# 1) 要跑的 parallel 清單
pars=(1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 30 40 50 70 100 196 384)

# 設定各路徑
SUBARRAY_CPP="../RT_SIM_IMC/src/SubArray.cpp"
RUN_SCRIPT="03_run_emb.sh"
BUILD_SCRIPT="02_build_gem5.sh"

# 迴圈跑每個平行度
for par in "${pars[@]}"; do
  echo "=== parallel = $par ==="

  # 2) 修改 SubArray.cpp 的 parallel 值
  sed -i -E "s#(static const int parallel\s*=\s*)[0-9]+;#\1${par};#" "$SUBARRAY_CPP"

  # 3) 執行 build（等到完成才繼續）
  echo "Building gem5 for parallel=$par..."
  ./"$BUILD_SCRIPT"
  if [ $? -ne 0 ]; then
    echo "  [Error] 建置失敗，停止腳本。"
    exit 1
  fi

  # 4) 修改 run script 的 outdir 參數
  #    把 --outdir=m5out_SVHN_224_p<數字> 換成當前 parallel
  sed -i -E "s#(--outdir=m5out_Flower_emb/patch_8/m5out_Flower_128_p)[0-9]+#\1${par}#" "$RUN_SCRIPT"

  # 5) 背景執行，log 依 parallel 命名
  LOGFILE="Flower_log/patch_8/gem5_Flower_128_p${par}.log"
  echo "Launching run script, log -> $LOGFILE"
  nohup ./"$RUN_SCRIPT" >& "$LOGFILE" &
  
  # 若不想太密集，也可以在這裡加個 sleep，例如：sleep 5
  sleep 5
done

echo "All jobs launched."
