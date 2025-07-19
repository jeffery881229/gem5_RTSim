#include <stdio.h>
#include <stdint.h>

#define MEMORY_SIZE 300000  // 主記憶體大小

void read_memory(uint8_t *memory, int n) {
    uint8_t reg = 0;  // 暫存器（單個 8-bit）

    for (int i = 0; i < n; i++) {
        reg = memory[i];  // 將記憶體資料存入暫存器
        // printf("Accessing memory[%d]: %02X, Register: %02X\n", i, memory[i], reg);
    }
}

int main() {
    // 模擬主記憶體（大小 1024 bytes）
    uint8_t main_memory[MEMORY_SIZE];

    // 初始化主記憶體，將其填滿一些假資料
    // for (int i = 0; i < MEMORY_SIZE; i++) {
    //     main_memory[i] = i & 0xFF;  // 每個地址存入 0x00 到 0xFF 循環值
    // }

    int n = 248832;  // 要存取的資料數量
    if (n > MEMORY_SIZE) {
        printf("Error: Requested data exceeds memory size.\n");
        return -1;
    }

    // 從主記憶體存取 n 筆資料到暫存器
    read_memory(main_memory, n);

    return 0;
}
