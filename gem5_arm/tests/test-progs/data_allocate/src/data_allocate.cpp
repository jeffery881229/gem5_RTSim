#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

int main() {
    // 假設要映射的物理地址是 0x10000000
    unsigned long phys_addr = 0x10000000;
    unsigned long page_size = sysconf(_SC_PAGESIZE); // 取得頁大小（一般為4KB）

    // 打開 /dev/mem 來訪問物理內存
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd == -1) {
        std::cerr << "無法打開 /dev/mem" << std::endl;
        return -1;
    }

    // 使用 mmap 將物理地址映射到虛擬地址空間
    void *mapped_base = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, phys_addr & ~(page_size - 1));
    if (mapped_base == MAP_FAILED) {
        std::cerr << "mmap 失敗" << std::endl;
        close(mem_fd);
        return -1;
    }

    // 計算物理地址的偏移量，因為 mmap 映射的是整個頁
    void *mapped_addr = (void *)((char *)mapped_base + (phys_addr & (page_size - 1)));

    // 將物理內存的值寫入數據
    *(volatile unsigned int *)mapped_addr = 42; // 將數據42寫入物理地址

    // 讀取該地址的數據
    unsigned int read_value = *(volatile unsigned int *)mapped_addr;
    std::cout << "讀取到的值為: " << read_value << std::endl;

    // 解除映射
    if (munmap(mapped_base, page_size) == -1) {
        std::cerr << "munmap 失敗" << std::endl;
    }

    // 關閉 /dev/mem 文件描述符
    close(mem_fd);

    return 0;
}
