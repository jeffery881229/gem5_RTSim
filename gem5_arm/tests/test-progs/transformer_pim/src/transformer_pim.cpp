#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  // for rand()
#include <cstring>
#include <ctime>    // for time()
#include <string>
#include <cstdint>  // for uint8_t, int8_t
#include <bitset>
// #include <x86intrin.h> // write through

// ---------------------- 追加或調整的巨集定義 ----------------------
#define Embe_PIM_ADDR 0xf0000000ULL      // virtual address of the Patch Embedding PIM

// Add quantization parameters
#define SCALE_FACTOR 127.0f
#define ZERO_POINT    0

// --------------------- ViT 參數 (可自行調整) ----------------------

// 假設我們的輸入影像大小: H x W x C
static const int imageH   = 32;   // 圖片高度 (示例)
static const int imageW   = 32;   // 圖片寬度 (示例)
static const int imageC   = 3;   // 圖片通道 (RGB)

// Patch 大小
static const int patchSize = 32;  // 每個 Patch 的寬與高

// 嵌入維度 (embedding dimension)
static const int dModel   = 384;   // Patch Embedding + Position Embedding 後的維度
// 96不會過，48會過

// 注意力頭數
static const int numHeads = 2;  
static const int dHead    = dModel / numHeads;  // 每個 head 的維度 (192)

// Feed Forward hidden dim (在 Encoder Block 內的 MLP)
static const int dFF      = 16;  


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

    // 1. 從記憶體讀取圖像
    uint8_t *image_data = (uint8_t *)malloc(imageH * imageW * imageC * sizeof(uint8_t));
    // 讀取已經 resize 好的 image data (.bin 格式)
    FILE *fp = fopen("data_set/SVHN/svhn_img0_32x32.bin", "rb");
    /*
    data_set/CIFAR-10/cifar10_img0_32x32.bin
    data_set/ImageNet_ILSVRC-2012/ILSVRC2012_val_00000024_32x32.bin
    data_set/Oxford_102_Flower/flower_img0_32x32.bin
    data_set/SVHN/svhn_img0_32x32.bin
    */
    if (!fp) {
        std::cerr << "[Error] Failed to open image file!\n";
        return -1;
    }

    size_t expected_size = imageH * imageW * imageC;
    size_t read_size = fread(image_data, sizeof(uint8_t), expected_size, fp);
    fclose(fp);

    if (read_size != expected_size) {
        std::cerr << "[Error] Image size mismatch: expected " << expected_size
                  << ", but got " << read_size << "\n";
        return -1;
    }

    std::cout << "[Info] Loaded image data successfully.\n";
    std::cout << "[Info] First 16 bits of image_data: ";
    std::bitset<8> byte1(image_data[0]);
    std::bitset<8> byte2(image_data[1]);
    std::cout << byte1 << " " << byte2 << std::endl;
    printf("image_data: %p\n", (void*)image_data);

    // 2. 準備 Patch Embedding 的權重
    int patchFlattenDim = patchSize * patchSize * imageC; 
    Matrix W_patch = createMatrix(patchFlattenDim, dModel, 0.0);
    std::vector<double> b_patch(dModel, 0.0);

    // 隨機初始化
    for(int i = 0; i < patchFlattenDim; i++){
        for(int j = 0; j < dModel; j++){
            W_patch[i][j] = ((double)(std::rand() % 100) / 50.0) - 1.0; 
        }
    }
    for(int j = 0; j < dModel; j++){
        b_patch[j] = 0.1; 
    }

    //****************************
    // 將 W_patch, b_patch寫入記憶體
    //****************************
    // 先量化成 int8
    // 改成用malloc來宣告，避免發生stack overflow
    int8_t* W_patch_quant = (int8_t*)malloc(patchFlattenDim * dModel * sizeof(int8_t));
    if (!W_patch_quant) {
        std::cerr << "Failed to allocate W_patch_quant" << std::endl;
        return -1;
    }
    int8_t* b_patch_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    if (!b_patch_quant) {
        std::cerr << "Failed to allocate b_patch_quant" << std::endl;
        free(W_patch_quant);
        return -1;
    }

    for(int i = 0; i < patchFlattenDim; i++) {
        for(int j = 0; j < dModel; j++) {
            W_patch_quant[i * dModel + j] = quantize(W_patch[i][j]);
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
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);


    // 3. CLS Token + Position Embedding
    std::vector<double> CLS_token(dModel, 0.0);
    // CLS token 初始化
    int8_t* CLS_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
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
    int8_t* posEmb_quant = (int8_t*)malloc((numPatches + 1) * dModel * sizeof(int8_t));
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
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

    // 4. 準備 Transformer Encoder (單層) 權重
    auto initMatrixRand = [&](Matrix &mat){
        for(int i = 0; i < (int)mat.size(); i++){
            for(int j = 0; j < (int)mat[i].size(); j++){
                mat[i][j] = ((double)(std::rand() % 100) / 100.0) - 0.5;
            }
        }
    };
    auto initVectorConst = [&](std::vector<double> &v, double c){
        for(int i = 0; i < (int)v.size(); i++){
            v[i] = c;
        }
    };

    Matrix Wq = createMatrix(dModel, dModel, 0.0);
    Matrix Wk = createMatrix(dModel, dModel, 0.0);
    Matrix Wv = createMatrix(dModel, dModel, 0.0);
    Matrix Wo = createMatrix(dModel, dModel, 0.0);
    std::vector<double> bq(dModel, 0.0), bk(dModel, 0.0);
    std::vector<double> bv(dModel, 0.0), bo(dModel, 0.0);

    initMatrixRand(Wq);
    initMatrixRand(Wk);
    initMatrixRand(Wv);
    initMatrixRand(Wo);
    initVectorConst(bq, 0.01);
    initVectorConst(bk, 0.01);
    initVectorConst(bv, 0.01);
    initVectorConst(bo, 0.01);

    Matrix W1 = createMatrix(dModel, dFF, 0.0);
    Matrix W2 = createMatrix(dFF, dModel, 0.0);
    std::vector<double> b1(dFF, 0.01), b2(dModel, 0.01);

    initMatrixRand(W1);
    initMatrixRand(W2);
    
    // 5. 執行單層 Encoder
    // Multi-Head Self-Attention
    // quantized Wq, Wk, Wv, Wo, bq, bk, bv, bo
    int8_t* Wq_quant = (int8_t*)malloc(dModel * dModel * sizeof(int8_t));
    int8_t* Wk_quant = (int8_t*)malloc(dModel * dModel * sizeof(int8_t));
    int8_t* Wv_quant = (int8_t*)malloc(dModel * dModel * sizeof(int8_t));
    int8_t* Wo_quant = (int8_t*)malloc(dModel * dModel * sizeof(int8_t));
    int8_t* bq_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    int8_t* bk_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    int8_t* bv_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    int8_t* bo_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    printf("Wq_quant: %p\n", (void*)Wq_quant);
    printf("Wk_quant: %p\n", (void*)Wk_quant);
    printf("Wv_quant: %p\n", (void*)Wv_quant);
    printf("Wo_quant: %p\n", (void*)Wo_quant);
    printf("bq_quant: %p\n", (void*)bq_quant);
    printf("bk_quant: %p\n", (void*)bk_quant);
    printf("bv_quant: %p\n", (void*)bv_quant);
    printf("bo_quant: %p\n", (void*)bo_quant);

    if (!Wq_quant || !Wk_quant || !Wv_quant || !Wo_quant || !bq_quant || !bk_quant || !bv_quant || !bo_quant) {
        std::cerr << "Failed to allocate Wqkvo_quant" << std::endl;
        return -1;
    }
    for(int i = 0; i < dModel; i++) {
        for(int j = 0; j < dModel; j++) {
            Wq_quant[i * dModel + j] = quantize(Wq[i][j]);
            Wk_quant[i * dModel + j] = quantize(Wk[i][j]);
            Wv_quant[i * dModel + j] = quantize(Wv[i][j]);
            Wo_quant[i * dModel + j] = quantize(Wo[i][j]);
        }
    }
    for(int i = 0; i < dModel; i++) {
        bq_quant[i] = quantize(bq[i]);
        bk_quant[i] = quantize(bk[i]);
        bv_quant[i] = quantize(bv[i]);
        bo_quant[i] = quantize(bo[i]);
    }


    //****************************
    // calculate Q, K, V in memory (PIM)
    //****************************
    *embedding_pim_addr = 0x12;
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

    // number of heads = 2
    // some declarations
    uint8_t *temp_vir_addr4 = (uint8_t *)malloc(sizeof(uint8_t));
    double *matrix_for_softmax = (double *)malloc(numPatches * numPatches * sizeof(double));
    double scale = 1.0 / std::sqrt((double)dHead); // 0.0721687836487032
    uint8_t *temp_vir_addr5 = (uint8_t *)malloc(sizeof(uint8_t));
    for(int k = 0; k < numHeads; k++) {
        /////////////////////////////////////////////
        /////// softmax(Q*KT/sqrt(dHead)) * V ///////
        /////////////////////////////////////////////
        //****************************
        // 1. calculate Q*KT in memory (PIM)
        //****************************
        *embedding_pim_addr = 0x13;
        printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

        // 2. read the result of Q*KT (numPatches,numPatches) from memory, (49, 49)
        //    and divided by sqrt(dHead)
        // use the memory location of weight matrix in patch embedding as the location of Q*KT
        // allocate another space for storing the double type result of divide by sqrt(dHead)
        for(int i = 0; i < numPatches; i++){
            for(int j = 0; j < numPatches; j++){
                matrix_for_softmax[i*numPatches + j] = (double)W_patch_quant[i*numPatches + j + k*dHead] * scale; // (Q*KT)/sqrt(dHead)
            }
        }
        // 3. do the softmax
        for(int i = 0; i < numPatches; i++){
            double maxVal = -1e9;
            for(int j = 0; j < numPatches; j++){
                if(matrix_for_softmax[i*numPatches + j] > maxVal) maxVal = matrix_for_softmax[i*numPatches + j];
            }
            double sumExp = 0.0;
            for(int j = 0; j < numPatches; j++){
                sumExp += std::exp(matrix_for_softmax[i*numPatches + j] - maxVal);
            }
            for(int j = 0; j < numPatches; j++){
                matrix_for_softmax[i*numPatches + j] = std::exp(matrix_for_softmax[i*numPatches + j] - maxVal) / sumExp;
            }
        }
        // quantize the softmax result
        for(int i = 0; i < numPatches; i++){
            for(int j = 0; j < numPatches; j++){
                W_patch_quant[i*numPatches + j + k*dHead] = quantize(matrix_for_softmax[i*numPatches + j]);
            }
        }

        //****************************
        // 4. multiply by V in memory (PIM)
        //****************************
        *embedding_pim_addr = 0x14;
        printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);
    }

    //****************************
    // result * Wo + bo (PIM)
    //****************************
    *embedding_pim_addr = 0x15;
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

    // Residual + LayerNorm
    //****************************
    // 1. add result with original (PIM)
    //****************************
    *embedding_pim_addr = 0x16;
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

    // 2. LayerNorm, Matrix size: (numPatches + 1, dModel)
    // use posEmb_quant as matrix to be processed, cause the size happens to be the same
    // malloc a double type of Matrix(numPatches + 1, dModel) to calculate LayerNorm
    double *layernormMatrix = (double *)malloc((numPatches + 1) * dModel * sizeof(double));
    for(int i = 0; i < numPatches + 1; i++){
        for(int j = 0; j < dModel; j++){
            layernormMatrix[i*dModel + j] = (double)posEmb_quant[i*dModel + j];
        }
    }
    // begin cal
    double eps = 1e-5;
    int rows = numPatches + 1;
    int cols = dModel;
    for(int i = 0; i < rows; i++){
        double mean = 0.0;
        for(int j = 0; j < cols; j++){
            mean += layernormMatrix[i*cols + j];
        }
        mean /= (double)cols;
        
        double var = 0.0;
        for(int j = 0; j < cols; j++){
            double diff = (layernormMatrix[i*cols + j] - mean);
            var += diff * diff;
        }
        var /= (double)cols;
        
        double inv = 1.0 / std::sqrt(var + eps);
        for(int j = 0; j < cols; j++){
            layernormMatrix[i*cols + j] = (layernormMatrix[i*cols + j] - mean) * inv;
        }
    }
    // restore the result
    for(int i = 0; i < numPatches + 1; i++){
        for(int j = 0; j < dModel; j++){
            posEmb_quant[i*dModel + j] = quantize(layernormMatrix[i*dModel + j]);
        }
    }
    
    // feed forward
    // max(0, XW1 + b1)W2 + b2 (PIM)
    // 1. quantized W1, W2, b1, b2 and send to memory
    int8_t* W1_quant = (int8_t*)malloc(dModel * dFF * sizeof(int8_t));
    int8_t* W2_quant = (int8_t*)malloc(dFF * dModel * sizeof(int8_t));
    int8_t* b1_quant = (int8_t*)malloc(dFF    * sizeof(int8_t));
    int8_t* b2_quant = (int8_t*)malloc(dModel * sizeof(int8_t));
    printf("W1_quant: %p\n", (void*)W1_quant);
    printf("W2_quant: %p\n", (void*)W2_quant);
    printf("b1_quant: %p\n", (void*)b1_quant);
    printf("b2_quant: %p\n", (void*)b2_quant);

    if (!W1_quant || !W2_quant || !b1_quant || !b2_quant) {
        std::cerr << "Failed to allocate W12_quant" << std::endl;
        return -1;
    }
    for(int i = 0; i < dModel; i++) {
        for(int j = 0; j < dFF; j++) {
            W1_quant[i * dFF + j] = quantize(W1[i][j]);
        }
    }
    for(int i = 0; i < dFF; i++) {
        for(int j = 0; j < dModel; j++) {
            W2_quant[i * dModel + j] = quantize(W2[i][j]);
        }
    }
    for(int i = 0; i < dFF; i++) {
        b1_quant[i] = quantize(b1[i]);
    }
    for(int i = 0; i < dModel; i++) {
        b2_quant[i] = quantize(b2[i]);
    }
    //****************************
    // 2. calculate max(0, XW1 + b1)W2 + b2 (PIM)
    //****************************
    *embedding_pim_addr = 0x17;
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

    // residual + LayerNorm
    //****************************
    // 1. add back to original result
    //****************************
    *embedding_pim_addr = 0x18;
    printf("Wrote to addr: %p, value: 0x%02x\n", (void*)embedding_pim_addr, *embedding_pim_addr);

    // 2. LayerNorm, Matrix size: (numPatches + 1, dModel)
    // use posEmb_quant as matrix to be processed, cause the size happens to be the same
    // malloc a double type of Matrix(numPatches + 1, dModel) to calculate LayerNorm
    double *layernormMatrix_2 = (double *)malloc((numPatches + 1) * dModel * sizeof(double));
    for(int i = 0; i < numPatches + 1; i++){
        for(int j = 0; j < dModel; j++){
            layernormMatrix_2[i*dModel + j] = (double)posEmb_quant[i*dModel + j];
        }
    }
    // begin cal
    for(int i = 0; i < rows; i++){
        double mean = 0.0;
        for(int j = 0; j < cols; j++){
            mean += layernormMatrix_2[i*cols + j];
        }
        mean /= (double)cols;
        
        double var = 0.0;
        for(int j = 0; j < cols; j++){
            double diff = (layernormMatrix_2[i*cols + j] - mean);
            var += diff * diff;
        }
        var /= (double)cols;
        
        double inv = 1.0 / std::sqrt(var + eps);
        for(int j = 0; j < cols; j++){
            layernormMatrix_2[i*cols + j] = (layernormMatrix_2[i*cols + j] - mean) * inv;
        }
    }
}
