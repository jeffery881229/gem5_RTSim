# nohup ./03_run.sh > gem5.log 2>&1 &
nohup ./03_run.sh >& gem5_SVHN_32_CPU3_pim.log &
nohup ./03_run_x86.sh >& gem5_SVHN_32_CPU1_pim.log &

# check: ps aux | grep gem5