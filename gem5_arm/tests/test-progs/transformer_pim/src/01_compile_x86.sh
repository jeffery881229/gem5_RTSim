# g++ -o ../bin/x86/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11
# arm-linux-gnueabihf-g++ -o ../bin/x86/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11
# arm-linux-gnueabihf-g++ -static -o ../bin/arm/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11

g++ -static-libstdc++ -static-libgcc \
    -O0 \
    -fno-builtin \
    -fno-tree-loop-distribute-patterns \
    -mno-avx -mno-avx2 \
    -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 \
    -std=c++11 \
    -pthread \
    -o ../bin/x86/linux/transformer_pim \
    transformer_pim.cpp