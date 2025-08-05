# g++ -o ../bin/x86/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11
# g++ -static-libstdc++ -static-libgcc -o ../bin/x86/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11

/usr/bin/g++ -static-libstdc++ -static-libgcc \
    -O0 \
    -fno-builtin \
    -fno-tree-loop-distribute-patterns \
    -mno-avx -mno-avx2 \
    -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 \
    -std=c++11 \
    -pthread \
    -o ../bin/x86/linux/embedding_pim_backup \
    embedding_pim_backup.cpp
