#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h> // For malloc()
#include <time.h>   // For rand()
#include <cmath> 

#define IMG_HEIGHT 224
#define IMG_WIDTH 224
#define PATCH_SIZE 64
#define NUM_CHANNELS 3
#define NUM_PATCHES ((IMG_HEIGHT / PATCH_SIZE) * (IMG_WIDTH / PATCH_SIZE)) // 196
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS) // 16 x 16 x 3 = 768
#define PROJ_DIM 384  // Projection dimension
#define INPUT_ADDR 0x100000000ULL      // Starting virtual address of the image

int main() {
    // Seed the random number generator
    srand(time(NULL));

    // Assume the image data is already in memory arranged by patches
    uint8_t *image_data = (uint8_t *)INPUT_ADDR;

    // Allocate memory for the projection matrix
    // float *projection_matrix = (float *)malloc(PROJ_DIM * PATCH_DIM * sizeof(float));
    // if (!projection_matrix) {
    //     printf("Error: Memory allocation failed for projection matrix.\n");
    //     return -1;
    // }

    // Initialize the projection matrix with random values between -1 and 1
    // for (int i = 0; i < PROJ_DIM * PATCH_DIM; i++) {
    //     projection_matrix[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    // }

    // Variable to store the projected vector
    float *projected_vector = (float *)malloc(sizeof(int16_t));
    if (!projected_vector) {
        printf("Error: Memory allocation failed for projected vector.\n");
        // free(projection_matrix);
        return -1;
    }

    int parallel = 16;
    // Process each patch
    for (int patch_idx = 0; patch_idx < std::ceil(NUM_PATCHES/parallel); patch_idx++) { // NUM_PATCHES = 196
        // Calculate the starting address of the current patch in the image data
        uint8_t *patch_data = image_data + patch_idx * PATCH_DIM * sizeof(uint8_t);

        // Perform linear projection: projected_vector = projection_matrix * patch_data
        // For each dimension in the projected vector
        for (int i = 0; i < PROJ_DIM; i++) {
            projected_vector[0] = 0.0f;
            // Multiply the i-th row of the projection matrix with the patch data
            for (int j = 0; j < PATCH_DIM; j++) {
                projected_vector[0] += (int16_t)patch_data[j] * 2;
            }
        }

        // Since we don't need to store the projected vector, we can reuse the same memory
        // Optionally, process the projected_vector here if needed
    }

    // Free allocated memory
    // free(projection_matrix);
    free(projected_vector);

    return 0;
}
