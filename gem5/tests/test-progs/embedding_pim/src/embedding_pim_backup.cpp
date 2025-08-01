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

// ---------------------- 追加或調整的巨集定義 ----------------------
#define INPUT_ADDR 0x100000000ULL  // Starting virtual address of the image
#define W_PATCH_ADDR 0x100025000ULL  // Starting virtual address of W_patch
// #define B_PATCH_ADDR 0x100144E00ULL  // Starting virtual address of b_patch
#define B_PATCH_ADDR 0x100146000ULL  // Starting virtual address of b_patch
#define CLS_ADDR 0x100147000ULL  // Starting virtual address of CLS token
#define POS_ADDR 0x100148000ULL  // Starting virtual address of Position Embedding
#define Wqkvo_ADDR 0x10014D000ULL  // Starting virtual address of four weight matrix Wq, Wk, Wv, Wo
#define Bqkvo_ADDR 0x1001DE000ULL  // Starting virtual address of four bias vector bq, bk, bv, bo
#define Embe_PIM_ADDR 0xf0000000ULL      // virtual address of the Patch Embedding PIM
#define CLS_POS_PIM_ADDR 0xf00001000ULL      // virtual address of the Patch Embedding PIM
#define QKV_cal_PIM_ADDR 0xf00002000ULL      // virtual address of the executing Q K V matrix PIM
#define Q_KT_cal_PIM_ADDR 0xf00003000ULL      // virtual address of the executing Q*KT matrix PIM
#define Soft_V_cal_PIM_ADDR 0xf00004000ULL      // virtual address of the executing softmax*V matrix PIM
#define Wo_bo_cal_PIM_ADDR 0xf00005000ULL      // virtual address of the executing Wo and bo matrix PIM
#define Add_back_PIM_ADDR 0xf00006000ULL      // virtual address of the executing add result back PIM
#define Feed_Forward_PIM_ADDR 0xf00007000ULL      // virtual address of the executing Feed Forward PIM
#define Add_back_PIM_ADDR_2 0xf00008000ULL      // virtual address of the executing add result back PIM 2
#define IMG_HEIGHT 224
#define IMG_WIDTH 224
#define NUM_CHANNELS 3

// Add quantization parameters
#define SCALE_FACTOR 127.0f
#define ZERO_POINT    0

// --------------------- ViT 參數 (可自行調整) ----------------------

// 假設我們的輸入影像大小: H x W x C
static const int imageH   = 32;   // 圖片高度 (示例)
static const int imageW   = 32;   // 圖片寬度 (示例)
static const int imageC   = 3;   // 圖片通道 (RGB)

// Patch 大小
static const int patchSize = 16;  // 每個 Patch 的寬與高

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

// 將 int8 反量化回 double
inline double dequantize(int8_t x) {
    return (double)(x - ZERO_POINT) / SCALE_FACTOR;
}

// --------------------- INT8 矩陣乘法 ----------------------
/*
   將 A, B (double) 先分別量化成 int8，再做 int8 乘法累加 (int32)，
   最後把結果再反量化回 double 存在 C
*/
Matrix matmulOffloadInt8(const Matrix &A, const Matrix &B) {
    int M = A.size();
    if (M == 0) return Matrix(); // edge case
    int K = A[0].size();

    int Kb = B.size();
    int N  = (Kb > 0) ? (int)B[0].size() : 0;
    // 檢查維度
    if (K != Kb) {
        std::cerr << "[Error] A.cols != B.rows in matmulOffloadInt8\n";
        return createMatrix(0,0,0.0);
    }

    // 建立輸出矩陣 C (double)
    Matrix C = createMatrix(M, N, 0.0);

    // 量化 A -> A_quant, B -> B_quant
    std::vector<std::vector<int8_t>> A_quant(M, std::vector<int8_t>(K, 0));
    std::vector<std::vector<int8_t>> B_quant(K, std::vector<int8_t>(N, 0));

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            A_quant[i][j] = quantize(A[i][j]);
        }
    }
    for(int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            B_quant[i][j] = quantize(B[i][j]);
        }
    }

    // INT8 x INT8 -> INT32 accumulation
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            int32_t acc = 0;
            for(int k = 0; k < K; k++) {
                acc += (int32_t)A_quant[i][k] * (int32_t)B_quant[k][j];
            }
            // 這裡簡單做「平均」再反量化，可依實際需求調整
            // (acc / K) -> int8 -> double
            double tmpDouble = dequantize( quantize( (double)acc / (double)K ) );
            C[i][j] = tmpDouble;
        }
    }

    return C;
}

// 加 bias 
void addBias(Matrix &mat, const std::vector<double> &bias) {
    // 假設 mat.shape = [rows, cols], bias.size = cols
    int rows = mat.size();
    if (rows == 0) return;
    int cols = mat[0].size();

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            mat[i][j] += bias[j];
        }
    }
}

// 對每個 row 做 softmax
void rowSoftmax(Matrix &mat) {
    for(int i = 0; i < (int)mat.size(); i++){
        double maxVal = -1e9;
        for(int j = 0; j < (int)mat[i].size(); j++){
            if(mat[i][j] > maxVal) maxVal = mat[i][j];
        }
        double sumExp = 0.0;
        for(int j = 0; j < (int)mat[i].size(); j++){
            sumExp += std::exp(mat[i][j] - maxVal);
        }
        for(int j = 0; j < (int)mat[i].size(); j++){
            mat[i][j] = std::exp(mat[i][j] - maxVal) / sumExp;
        }
    }
}

// LayerNorm 
void layerNormInPlace(Matrix &mat, double eps = 1e-5) {
    int rows = mat.size();
    if(rows == 0) return;
    int cols = mat[0].size();
    
    for(int i = 0; i < rows; i++){
        double mean = 0.0;
        for(int j = 0; j < cols; j++){
            mean += mat[i][j];
        }
        mean /= (double)cols;
        
        double var = 0.0;
        for(int j = 0; j < cols; j++){
            double diff = (mat[i][j] - mean);
            var += diff * diff;
        }
        var /= (double)cols;
        
        double inv = 1.0 / std::sqrt(var + eps);
        for(int j = 0; j < cols; j++){
            mat[i][j] = (mat[i][j] - mean) * inv;
        }
    }
}

// --------------------- Patch Embedding ----------------------
Matrix patchEmbedding(
    uint8_t* image_data,
    int imgH, int imgW, int imgC,
    int pSize,
    const Matrix &W_patch,
    const std::vector<double> &b_patch
) {
    int patchesY = imgH / pSize;
    int patchesX = imgW / pSize;
    int numPatches = patchesY * patchesX;
    int patchFlattenDim = pSize * pSize * imgC;

    Matrix tokens = createMatrix(numPatches, dModel, 0.0);

    int patchIdx = 0;
    // 對每個 patch
    for(int py = 0; py < patchesY; py++){
        for(int px = 0; px < patchesX; px++){

            // 準備 flattenPatch (1 x patchFlattenDim), double
            Matrix flattenPatch = createMatrix(1, patchFlattenDim, 0.0);
            int flatPos = 0;  // 每個 patch 都要從 0 開始
            for(int i = 0; i < pSize; i++){
                for(int j = 0; j < pSize; j++){
                    for(int c = 0; c < imgC; c++){
                        int srcY = py * pSize + i;
                        int srcX = px * pSize + j;
                        int idx = (srcY * imgW * imgC) + (srcX * imgC) + c;
                        flattenPatch[0][flatPos++] = static_cast<double>(image_data[idx]);
                    }
                }
            }

            // (1 x patchFlattenDim) x (patchFlattenDim x dModel)
            Matrix proj = matmulOffloadInt8(flattenPatch, W_patch);
            addBias(proj, b_patch);

            // 放入 tokens[patchIdx]
            for(int d = 0; d < dModel; d++){
                tokens[patchIdx][d] = proj[0][d];
            }
            patchIdx++;
        }
    }
    return tokens;
}

// --------------------- Multi-Head Self-Attention ----------------------
Matrix multiHeadSelfAttention(
    const Matrix &X,   // (numPatches + 1, dModel)
    const Matrix &Wq,
    const Matrix &Wk,
    const Matrix &Wv,
    const Matrix &Wo,
    const std::vector<double> &bq,
    const std::vector<double> &bk,
    const std::vector<double> &bv,
    const std::vector<double> &bo
){
    int nTokens = X.size();
    // Q, K, V
    Matrix Q = matmulOffloadInt8(X, Wq);
    addBias(Q, bq);

    Matrix K = matmulOffloadInt8(X, Wk);
    addBias(K, bk);

    Matrix V = matmulOffloadInt8(X, Wv);
    addBias(V, bv);

    // 分多頭
    Matrix outAll = createMatrix(nTokens, dModel, 0.0);

    for(int h = 0; h < numHeads; h++){
        // Qh, Kh, Vh = (nTokens, dHead)
        Matrix Qh = createMatrix(nTokens, dHead, 0.0);
        Matrix Kh = createMatrix(nTokens, dHead, 0.0);
        Matrix Vh = createMatrix(nTokens, dHead, 0.0);

        for(int i = 0; i < nTokens; i++){
            for(int j = 0; j < dHead; j++){
                Qh[i][j] = Q[i][h*dHead + j];
                Kh[i][j] = K[i][h*dHead + j];
                Vh[i][j] = V[i][h*dHead + j];
            }
        }

        // scores = Qh x Kh^T / sqrt(dHead)
        // Qh: (nTokens, dHead), Kh^T: (dHead, nTokens)
        // matmulOffloadInt8 expects (Matrix, Matrix)
        // 先轉 Kh => KhT
        Matrix KhT = createMatrix(dHead, nTokens);
        for(int i = 0; i < nTokens; i++){
            for(int j = 0; j < dHead; j++){
                KhT[j][i] = Kh[i][j];
            }
        }

        Matrix scores = matmulOffloadInt8(Qh, KhT);
        double scale = 1.0 / std::sqrt((double)dHead);
        for(int i = 0; i < nTokens; i++){
            for(int j = 0; j < nTokens; j++){
                scores[i][j] *= scale;
            }
        }
        rowSoftmax(scores);

        // context = scores x Vh
        Matrix context = matmulOffloadInt8(scores, Vh);

        // 放回 outAll
        for(int i = 0; i < nTokens; i++){
            for(int j = 0; j < dHead; j++){
                outAll[i][h*dHead + j] = context[i][j];
            }
        }
    }

    // 乘 Wo
    Matrix out = matmulOffloadInt8(outAll, Wo);
    addBias(out, bo);

    return out;
}

// --------------------- Feed Forward ----------------------
Matrix feedForward(
    const Matrix &X,
    const Matrix &W1,
    const Matrix &W2,
    const std::vector<double> &b1,
    const std::vector<double> &b2
){
    Matrix hidden = matmulOffloadInt8(X, W1);
    addBias(hidden, b1);

    // ReLU
    for(int i = 0; i < (int)hidden.size(); i++){
        for(int j = 0; j < (int)hidden[i].size(); j++){
            hidden[i][j] = std::max(0.0, hidden[i][j]);
        }
    }

    Matrix out = matmulOffloadInt8(hidden, W2);
    addBias(out, b2);
    return out;
}

// --------------------- Transformer Encoder ----------------------
Matrix transformerEncoderLayer(
    const Matrix &X,
    // MHA 權重
    const Matrix &Wq, const Matrix &Wk, const Matrix &Wv, const Matrix &Wo,
    const std::vector<double> &bq, const std::vector<double> &bk,
    const std::vector<double> &bv, const std::vector<double> &bo,
    // FFN 權重
    const Matrix &W1, const Matrix &W2,
    const std::vector<double> &b1, const std::vector<double> &b2
){
    // 1) MHA
    Matrix attnOut = multiHeadSelfAttention(X, Wq, Wk, Wv, Wo, bq, bk, bv, bo);

    // 2) Residual + LayerNorm
    Matrix X_attn = createMatrix(X.size(), dModel, 0.0);
    for(int i = 0; i < (int)X.size(); i++){
        for(int j = 0; j < dModel; j++){
            X_attn[i][j] = X[i][j] + attnOut[i][j];
        }
    }
    layerNormInPlace(X_attn);

    // 3) Feed Forward
    Matrix ffnOut = feedForward(X_attn, W1, W2, b1, b2);

    // 4) Residual + LayerNorm
    Matrix X_out = createMatrix(X_attn.size(), dModel, 0.0);
    for(int i = 0; i < (int)X_attn.size(); i++){
        for(int j = 0; j < dModel; j++){
            X_out[i][j] = X_attn[i][j] + ffnOut[i][j];
        }
    }
    layerNormInPlace(X_out);

    return X_out;
}

// --------------------- 主程式範例 ----------------------
int main(){
    std::srand((unsigned)std::time(nullptr));

    // 1. 從記憶體讀取圖像
    uint8_t *image_data = (uint8_t *)malloc(imageH * imageW * imageC * sizeof(uint8_t));
    // 讀取已經 resize 好的 image data (.bin 格式)
    FILE *fp = fopen("/data_set/CIFAR-10/cifar_img0_32x32.bin", "rb");
    /*
    /data_set/CIFAR-10/cifar10_img0_32x32.bin
    /data_set/ImageNet_ILSVRC-2012/ILSVRC2012_val_00000024_64x64.bin
    /data_set/Oxford_102_Flower/flower_img0_128x128.bin
    /data_set/SVHN/svhn_img0_224x224.bin
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
            int r = std::rand() % 256 - 128;
            W_patch_quant[i * dModel + j] = static_cast<int8_t>(r);
        }
    }
    printf("W_patch_quant: %p\n", (void*)W_patch_quant);
    
    for(int j = 0; j < dModel; j++) {
        b_patch_quant[j] = quantize(b_patch[j]);
    }
    printf("b_patch_quant: %p\n", (void*)b_patch_quant);


/*

    // 寫入記憶體
    uint8_t *w_patch_in_memory = (uint8_t *)W_PATCH_ADDR; 
    int total_memcpy_size = patchFlattenDim * dModel;
    // int memcpy_size = 46743; // 46743, 46742
    // 386783800496: system.cpu: 0x30698    : ud2, solution: add command when compiling
    int memcpy_size = 46767; // 46768, 46767
    for(int i=0; i<total_memcpy_size/memcpy_size; i++){
        for(int j=0; j<memcpy_size/16511; j++){
            memcpy(w_patch_in_memory + j*16511, W_patch_quant + i*memcpy_size + j*16511, 16511); // size max = 16511
        }
        if(memcpy_size%16511 != 0){
            memcpy(w_patch_in_memory + memcpy_size/16511*16511, W_patch_quant + i*memcpy_size + memcpy_size/16511*16511, memcpy_size%16511);
        }
    }
    if(total_memcpy_size%memcpy_size != 0){
        memcpy(w_patch_in_memory, W_patch_quant + total_memcpy_size/memcpy_size*memcpy_size, total_memcpy_size%memcpy_size);
    }
    // memcpy(w_patch_in_memory, W_patch_quant, patchFlattenDim*dModel); // size max = 16511
    free(W_patch_quant);

    uint8_t *b_patch_in_memory = (uint8_t *)B_PATCH_ADDR;
    memcpy(b_patch_in_memory, b_patch_quant, dModel);
    free(b_patch_quant);

    printf("finish memcpy\n");
*/

    // 進行 Patch Embedding (PIM)
    uint8_t *embedding_pim_addr = (uint8_t *)Embe_PIM_ADDR;
    *embedding_pim_addr = 0x10;
    // uint8_t *temp_vir_addr = (uint8_t *)malloc(sizeof(uint8_t));
    // temp_vir_addr[0] = *embedding_pim_addr;

    // Matrix patchTokens = patchEmbedding(image_data, imageH, imageW, imageC,
    //                                     patchSize,
    //                                     W_patch, b_patch);
    // int numPatches = patchTokens.size();


    // 3. CLS Token + Position Embedding
    // Matrix tokens = createMatrix(numPatches + 1, dModel, 0.0);
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

/*
    // send CLS token to memory
    uint8_t *cls_token_in_memory = (uint8_t *)CLS_ADDR;
    memcpy(cls_token_in_memory, CLS_quant, dModel);
*/

    // // 複製 patchTokens
    // for(int i = 0; i < numPatches; i++){
    //     for(int j = 0; j < dModel; j++){
    //         tokens[i+1][j] = patchTokens[i][j];
    //     }
    // }
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

/*
    // send Position Embedding to memory
    uint8_t *posEmb_in_memory = (uint8_t *)(POS_ADDR);
    for(int i=0; i<(numPatches + 1)*dModel/16511; i++){
        memcpy(posEmb_in_memory + i*16511, posEmb_quant + i*16511, 16511); // size max = 16511
    }
    if((numPatches + 1)*dModel%16511 != 0){
        memcpy(posEmb_in_memory + (numPatches + 1)*dModel/16511*16511, posEmb_quant + (numPatches + 1)*dModel/16511*16511, (numPatches + 1)*dModel%16511);
    }

*/
    
    // for(int i = 0; i < (numPatches + 1); i++){
    //     for(int d = 0; d < dModel; d++){
    //         tokens[i][d] += posEmb[i][d];
    //     }
    // }

    // do CLS token + Position Embedding in memory (PIM)
    *embedding_pim_addr = 0x11;
    // uint8_t *cls_pos_pim_addr = (uint8_t *)CLS_POS_PIM_ADDR;
    // uint8_t *temp_vir_addr2 = (uint8_t *)malloc(sizeof(uint8_t));
    // temp_vir_addr2[0] = *cls_pos_pim_addr;


    // print the result
    // for(int i = 0; i < (numPatches + 1); i++){
    //     for(int j = 0; j < dModel; j++){
    //         std::cout << static_cast<int>(posEmb_quant[i*dModel + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

}
