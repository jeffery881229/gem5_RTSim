#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  // for rand()
#include <ctime>    // for time()
#include <string>
#include <cstdint>  // for uint8_t, int8_t
#include <bitset>

// ---------------------- 追加或調整的巨集定義 ----------------------
// #define INPUT_ADDR 0x100000000ULL  // Starting virtual address of the image
// #define IMG_HEIGHT 64
// #define IMG_WIDTH 64
// #define NUM_CHANNELS 3

// Add quantization parameters
#define SCALE_FACTOR 127.0f
#define ZERO_POINT    0

// --------------------- ViT 參數 (可自行調整) ----------------------

// 假設我們的輸入影像大小: H x W x C
static const int imageH   = 224;   // 圖片高度 (示例)
static const int imageW   = 224;   // 圖片寬度 (示例)
static const int imageC   = 3;   // 圖片通道 (RGB)

// Patch 大小
static const int patchSize = 32;  // 每個 Patch 的寬與高

// 嵌入維度 (embedding dimension)
static const int dModel   = 384;   // Patch Embedding + Position Embedding 後的維度

// 注意力頭數
static const int numHeads = 2;  
static const int dHead    = dModel / numHeads;  // 每個 head 的維度

// Feed Forward hidden dim (在 Encoder Block 內的 MLP)
static const int dFF      = 16;  

// --------------------- 型別定義與工具函式 ----------------------

// 使用 double 作為儲存資料的型別
using Matrix = std::vector<std::vector<double>>;

// 建立 rows x cols 的 double 矩陣
Matrix createMatrix(int rows, int cols, double val = 0.0) {
    return Matrix(rows, std::vector<double>(cols, val));
}

// 印出矩陣 (for debug)
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
   最後把結果再反量化回 double 存在 C。
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


// 加上 bias (broadcast)
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
    const Matrix &X,   // (nTokens, dModel)
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

// --------------------- Feed Forward (MLP) ----------------------
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

// --------------------- 單層 Transformer Encoder (含 LayerNorm) ----------------------
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


int main(){
    std::srand((unsigned)std::time(nullptr));

    // 1. 從記憶體讀取圖像
    uint8_t *image_data = (uint8_t *)malloc(imageH * imageW * imageC * sizeof(uint8_t));
    // 讀取已經 resize 好的 image data (.bin 格式)
    FILE *fp = fopen("/RAID2/LAB/css/cssRA01/gem5/data_set/SVHN/svhn_img0_224x224.bin", "rb");
    /*
    /RAID2/LAB/css/cssRA01/gem5/data_set/CIFAR-10/cifar10_img0_224x224.bin
    /RAID2/LAB/css/cssRA01/gem5/data_set/ImageNet_ILSVRC-2012/ILSVRC2012_val_00000024_224x224.bin
    /RAID2/LAB/css/cssRA01/gem5/data_set/Oxford_102_Flower/flower_img0_224x224.bin
    /RAID2/LAB/css/cssRA01/gem5/data_set/SVHN/svhn_img0_224x224.bin
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

    // 產生 Patch Token
    Matrix patchTokens = patchEmbedding(image_data, imageH, imageW, imageC,
                                        patchSize,
                                        W_patch, b_patch);
    int numPatches = patchTokens.size();

    // 3. CLS Token + Position Embedding
    Matrix tokens = createMatrix(numPatches + 1, dModel, 0.0);
    // CLS token 初始化
    for(int j = 0; j < dModel; j++){
        tokens[0][j] = ((double)(std::rand() % 100) / 100.0);
    }
    // 複製 patchTokens
    for(int i = 0; i < numPatches; i++){
        for(int j = 0; j < dModel; j++){
            tokens[i+1][j] = patchTokens[i][j];
        }
    }
    // Position Embedding (簡單隨機加)
    Matrix posEmb = createMatrix(numPatches + 1, dModel, 0.0);
    for(int i = 0; i < (numPatches + 1); i++){
        for(int d = 0; d < dModel; d++){
            posEmb[i][d] = ((double)(std::rand() % 100) / 100.0) - 0.5;
        }
    }
    for(int i = 0; i < (numPatches + 1); i++){
        for(int d = 0; d < dModel; d++){
            tokens[i][d] += posEmb[i][d];
        }
    }

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
    Matrix encoderOut = transformerEncoderLayer(tokens,
                                                Wq, Wk, Wv, Wo,
                                                bq, bk, bv, bo,
                                                W1, W2, b1, b2);

    // 6. 取 CLS Token 作為輸出
    Matrix clsToken = createMatrix(1, dModel, 0.0);
    for(int j = 0; j < dModel; j++){
        clsToken[0][j] = encoderOut[0][j];
    }

    printMatrix(clsToken, "CLS Token Output");

    return 0;
}
