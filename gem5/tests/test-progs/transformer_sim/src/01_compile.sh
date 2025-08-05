# g++ -std=c++11 -O2 -pthread \
#     -I/RAID2/LAB/css/cssRA01/gem5/include \
#     transformer_sim.cpp \
#     /RAID2/LAB/css/cssRA01/gem5/util/m5/src/abi/x86/m5op.S \
#     -o ../bin/x86/linux/transformer_sim

g++ -o ../bin/x86/linux/transformer_sim_2 transformer_sim.cpp -pthread -std=c++11
# /usr/bin/g++ -o ../bin/x86/linux/transformer_sim_2 transformer_sim.cpp -pthread -std=c++11