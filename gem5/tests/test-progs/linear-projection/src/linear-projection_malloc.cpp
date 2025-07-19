#include <iostream>
#include <iomanip>
#include <cstdlib>  // 包含 malloc, free

using namespace std;

int main() {
    // 動態分配 inputVector 的記憶體 (3 個 int)
    int* inputVector = (int*)malloc(3 * sizeof(int));
    inputVector[0] = 123;
    inputVector[1] = 234;
    inputVector[2] = 12;

    // 動態分配 weightMatrix 的記憶體 (3x2 的矩陣)
    float** weightMatrix = (float**)malloc(3 * sizeof(float*));
    for (int i = 0; i < 3; ++i) {
        weightMatrix[i] = (float*)malloc(2 * sizeof(float));
    }
    weightMatrix[0][0] = 0.2f;
    weightMatrix[0][1] = 0.5f;
    weightMatrix[1][0] = 0.3f;
    weightMatrix[1][1] = 0.3f;
    weightMatrix[2][0] = 0.5f;
    weightMatrix[2][1] = 0.2f;

    // 動態分配 result 的記憶體 (2 個 float)
    float* result = (float*)malloc(2 * sizeof(float));
    result[0] = 0.0f;
    result[1] = 0.0f;

    // 計算結果
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 3; ++i) {
            result[j] += inputVector[i] * weightMatrix[i][j];
        }
    }

    // 輸出結果
    cout << "Resultant vector (1x2):\n";
    for (int j = 0; j < 2; ++j) {
        cout << fixed << setprecision(4) << result[j] << " ";
    }
    cout << endl;

    // 釋放動態分配的記憶體
    free(inputVector);
    for (int i = 0; i < 3; ++i) {
        free(weightMatrix[i]);
    }
    free(weightMatrix);
    free(result);

    return 0;
}
