#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  // for rand()
#include <cstring>
#include <ctime>    // for time()
#include <string>
#include <cstdint>  // for uint8_t, int8_t
#include <x86intrin.h> // write through
#include <bitset>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>

// ---------------------- 追加或調整的巨集定義 ----------------------
#define Embe_PIM_ADDR 0xf0000000ULL      // virtual address of the Patch Embedding PIM

// Add quantization parameters
#define SCALE_FACTOR 127.0f
#define ZERO_POINT    0

// --------------------- ViT 參數 (可自行調整) ----------------------

// 假設我們的輸入影像大小: H x W x C
static const int imageH   = 224;   // 圖片高度 (示例)
static const int imageW   = 224;   // 圖片寬度 (示例)
static const int imageC   = 3;   // 圖片通道 (RGB)

// Patch 大小
static const int patchSize = 64;  // 每個 Patch 的寬與高

// 嵌入維度 (embedding dimension)
static const int dModel   = 384;   // Patch Embedding + Position Embedding 後的維度
// 96不會過，48會過

// 注意力頭數
static const int numHeads = 2;  
static const int dHead    = dModel / numHeads;  // 每個 head 的維度 (192)

// Feed Forward hidden dim (在 Encoder Block 內的 MLP)
static const int dFF      = 16;  

// Example: allocate one page (4 KB) that is guaranteed page‐aligned
uint8_t *allocate_page_aligned_buffer(size_t bytes) {
    void *ptr = nullptr;
    int rc = posix_memalign(&ptr, 0x1000, bytes);
    if (rc != 0) {
        fprintf(stderr, "posix_memalign failed: %s\n", strerror(rc));
        return nullptr;
    }
    return (uint8_t*)ptr;
}

// --------------------- 型別定義與工具函式 ----------------------

// 使用 double 作為儲存資料的型別
using Matrix = std::vector<std::vector<double>>;

// 建立 rows x cols 的 double 矩陣
Matrix createMatrix(int rows, int cols, double val = 0.0) {
    return Matrix(rows, std::vector<double>(cols, val));
}

// 簡易印出矩陣 (for debug)
void printMatrix(const Matrix &mat, const std::string &name = "") {
    if (!name.empty()) {
        std::cout << name << ":\n";
    }
    for (int i = 0; i < (int)mat.size(); ++i) {
        for (int j = 0; j < (int)mat[i].size(); ++j) {
            std::cout << mat[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// --------------------- 量化 / 反量化 ----------------------

// 將 double 量化到 int8
inline int8_t quantize(double x) {
    // x * scale + zero_point
    double q = x * SCALE_FACTOR + ZERO_POINT;
    // clamp 到 int8 範圍
    if (q > 127.0)  q = 127.0;
    if (q < -128.0) q = -128.0;
    return static_cast<int8_t>(q);
}

// --------------------- 主程式範例 ----------------------
int main(){
    std::srand((unsigned)std::time(nullptr));

    const int numImages = 1;

    size_t imageSize = imageH * imageW * imageC;
    // 1. 準備五個指標，各自 malloc 一塊 buffer
    uint8_t *image_data[numImages];
    for (int i = 0; i < numImages; i++) {
        image_data[i] = (uint8_t *)malloc(imageSize * sizeof(uint8_t));
        if (!image_data[i]) {
            std::cerr << "[Error] 預留 image_data[" << i << "] 失敗！\n";
            // 如果有前面 i-1 張已經 malloc，要先 free 掉
            for (int j = 0; j < i; j++) {
                free(image_data[j]);
            }
            return -1;
        }
    }

    // 2. 依序打開五個檔案，把資料讀到對應的 buffer
    char filename[256];
    FILE *fp = nullptr;
    for (int idx = 0; idx < numImages; idx++) {
        // 組出檔名，例如 "/…/svhn_img0_224x224.bin"
        std::snprintf(filename, sizeof(filename), "data_set/SVHN/svhn_img%d_224x224.bin", idx);        
        /*
        data_set/CIFAR-10/cifar_img0_32x32.bin
        data_set/ImageNet_ILSVRC-2012/imagenet_img0_64x64.bin
        data_set/Oxford_102_Flower/flower_img0_128x128.bin
        data_set/SVHN/svhn_img0_224x224.bin
        */
        fp = std::fopen(filename, "rb");
        if (!fp) {
            std::cerr << "[Error] 無法開檔：" << filename << "\n";
            // 釋放之前 malloc 的 memory
            for (int j = 0; j < numImages; j++) {
                free(image_data[j]);
            }
            return -1;
        }

        size_t readBytes = std::fread(image_data[idx], 1, imageSize, fp);
        std::fclose(fp);
        if (readBytes != imageSize) {
            std::cerr << "[Error] 讀檔大小不符合，檔案：" << filename
                      << " 期待 " << imageSize << " bytes，但實際讀到 "
                      << readBytes << " bytes\n";
            for (int j = 0; j < numImages; j++) {
                free(image_data[j]);
            }
            return -1;
        }
        std::cout << "[Info] 成功讀取 " << filename << " 到 image_data[" << idx << "]\n";
    }

    // 2. 準備 Patch Embedding 的權重
    int patchFlattenDim = patchSize * patchSize * imageC; 
    // Matrix W_patch = createMatrix(patchFlattenDim, dModel, 0.0);
    std::vector<double> b_patch(dModel, 0.0);

    // 隨機初始化
    // for(int i = 0; i < patchFlattenDim; i++){
    //     for(int j = 0; j < dModel; j++){
    //         W_patch[i][j] = ((double)(std::rand() % 100) / 50.0) - 1.0; 
    //     }
    // }
    for(int j = 0; j < dModel; j++){
        b_patch[j] = 0.1; 
    }

    //****************************************************************
    // note: 在 process.cc 中使用map()來將記憶體映射到虛擬記憶體
    //       flags 設為 EmulationPageTable::Uncacheable 就不會出現問題了
    //****************************************************************
    //****************************
    // 將 W_patch, b_patch寫入記憶體
    //****************************
    // 先量化成 int8
    // 改成用malloc來宣告，避免發生stack overflow

    size_t bytes = patchFlattenDim * dModel * sizeof(int8_t); 
    // round up to a multiple of 0x1000 (the page size):
    size_t pages = (bytes + 0xFFF) & ~0xFFF;  
    uint8_t *W_patch_quant = allocate_page_aligned_buffer(pages);
    // int8_t* W_patch_quant = (int8_t*)malloc(patchFlattenDim * dModel * sizeof(int8_t));
    if (!W_patch_quant) {
        std::cerr << "Failed to allocate W_patch_quant" << std::endl;
        return -1;
    }
    bytes = dModel * sizeof(int8_t); 
    // round up to a multiple of 0x1000 (the page size):
    pages = (bytes + 0xFFF) & ~0xFFF;  
    uint8_t *b_patch_quant = allocate_page_aligned_buffer(pages);
    // int8_t* b_patch_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    if (!b_patch_quant) {
        std::cerr << "Failed to allocate b_patch_quant" << std::endl;
        free(W_patch_quant);
        return -1;
    }

    for(int i = 0; i < patchFlattenDim; i++) {
        for(int j = 0; j < dModel; j++) {
            int r = std::rand() % 256 - 128;
            W_patch_quant[i * dModel + j] = static_cast<int8_t>(r);
        }
    }
    printf("W_patch_quant: %p\n", (void*)W_patch_quant);
    //****************************************************************
    // note: 將 print 出來的 virtual address
    //       在 page_table.cc 中的 map()
    //       將 flags 設為 EmulationPageTable::Uncacheable 
    //****************************************************************
    
    for(int j = 0; j < dModel; j++) {
        b_patch_quant[j] = quantize(b_patch[j]);
    }
    printf("b_patch_quant: %p\n", (void*)b_patch_quant);
    
    //****************************
    // 進行 Patch Embedding (PIM)
    //****************************
    uint8_t *embedding_pim_addr = (uint8_t *)Embe_PIM_ADDR;
    *embedding_pim_addr = 0x10;

    // 3. CLS Token + Position Embedding
    std::vector<double> CLS_token(dModel, 0.0);
    // CLS token 初始化
    bytes = dModel * sizeof(int8_t); 
    // round up to a multiple of 0x1000 (the page size):
    pages = (bytes + 0xFFF) & ~0xFFF;  
    uint8_t *CLS_quant = allocate_page_aligned_buffer(pages);
    // int8_t* CLS_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    if (!CLS_quant) {
        std::cerr << "Failed to allocate CLS_quant" << std::endl;
        free(CLS_quant);
        return -1;
    }
    for(int j = 0; j < dModel; j++){
        CLS_token[j] = ((double)(std::rand() % 100) / 100.0);
    }
    for(int j = 0; j < dModel; j++){
        CLS_quant[j] = quantize(CLS_token[j]);
    }
    printf("CLS_quant: %p\n", (void*)CLS_quant);

    // Position Embedding (簡單隨機加)
    int numPatches = (imageH/patchSize) * (imageW/patchSize); // 49
    Matrix posEmb = createMatrix(numPatches + 1, dModel, 0.0);
    for(int i = 0; i < (numPatches + 1); i++){
        for(int d = 0; d < dModel; d++){
            posEmb[i][d] = ((double)(std::rand() % 100) / 100.0) - 0.5;
        }
    }
    bytes = (numPatches + 1) * dModel * sizeof(int8_t);
    // round up to a multiple of 0x1000 (the page size):
    pages = (bytes + 0xFFF) & ~0xFFF;  
    uint8_t *posEmb_quant = allocate_page_aligned_buffer(pages);
    // int8_t* posEmb_quant = (int8_t*)malloc((numPatches + 1) * dModel * sizeof(int8_t));
    if (!posEmb_quant) {
        std::cerr << "Failed to allocate posEmb_quant" << std::endl;
        free(posEmb_quant);
        return -1;
    }
    for(int i = 0; i < (numPatches + 1); i++){
        for(int j = 0; j < dModel; j++){
            posEmb_quant[i*dModel + j] = quantize(posEmb[i][j]);
        }
    }
    printf("posEmb_quant: %p\n", (void*)posEmb_quant);

    //****************************
    // do CLS token + Position Embedding in memory (PIM)
    //****************************
    *embedding_pim_addr = 0x11;

    // read the result (make sure all of the data were being read by CPU)
    uint64_t checksum = 0;
    for(int i = 0; i < (numPatches + 1); i++){
        for(int j = 0; j < dModel; j++){
            checksum += static_cast<int>(posEmb_quant[i*dModel + j]);
        }
    }
    std::cout << "[Info] Read back PIM result, checksum = 0x"
            << std::hex << checksum << std::dec << std::endl;

}