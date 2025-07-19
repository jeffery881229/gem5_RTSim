# nohup ./03_run.sh > gem5.log 2>&1 &
nohup ./03_run.sh >& log_emb_2/SVHN/patch_64/gem5_SVHN_224_baseline.log &
nohup ./03_run.sh >& gem5_m5op_12_layer_128img_baseline.log &
nohup ./03_run_emb.sh >& log_emb_2/SVHN/patch_8/gem5_SVHN_224.log &

nohup ./06_run_all_parallel.sh >& run_all_parallel_patch_8_Flower.log &

nohup ./07_run_all_emb.sh >& run_all_emb_new.log &

# check: ps aux | grep gem5
