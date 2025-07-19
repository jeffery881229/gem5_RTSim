#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h> // For rand()
#include <time.h>   // For seeding rand()

#define IMG_HEIGHT 224
#define IMG_WIDTH 224
#define PATCH_SIZE 16
#define NUM_CHANNELS 3
#define NUM_PATCHES ((IMG_HEIGHT / PATCH_SIZE) * (IMG_WIDTH / PATCH_SIZE)) // 196
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS) // 16 x 16 x 3 = 768
#define PROJ_DIM 384  // Projection dimension (adjusted as needed)
#define INPUT_ADDR 0x100000000ULL      // Starting virtual address of the image
#define IMAGE_SIZE (IMG_HEIGHT * IMG_WIDTH * NUM_CHANNELS) // 150,528 bytes

// Calculate sizes of output data and projection matrix
#define OUTPUT_SIZE (NUM_PATCHES * PROJ_DIM * sizeof(float)) // Output data size
#define PROJ_MATRIX_SIZE (PROJ_DIM * PATCH_DIM * sizeof(float)) // Projection matrix size

// Calculate virtual addresses for output and projection matrix, ensure they are within mapped range
#define OUTPUT_ADDR (INPUT_ADDR + IMAGE_SIZE) // Starting virtual address of the output data
#define PROJ_MATRIX_ADDR (OUTPUT_ADDR + OUTPUT_SIZE) // Starting virtual address of the projection matrix

int main() {
    // Verify all addresses are within the mapped virtual address range
    if (PROJ_MATRIX_ADDR + PROJ_MATRIX_SIZE > 0x120000000ULL) {
        // printf("Error: Memory addresses exceed mapped virtual address range.\n");
        return -1;
    }

    // Pointers to the memory-mapped regions
    uint8_t *img_data = (uint8_t *)INPUT_ADDR;         // Input image data
    // float *output_data = (float *)OUTPUT_ADDR;         // Output data after projection
    // float *projection_matrix = (float *)PROJ_MATRIX_ADDR; // Projection matrix
    float *output_data = (float *)malloc(OUTPUT_SIZE);
    float *projection_matrix = (float *)malloc(PROJ_MATRIX_SIZE);
    if (!output_data || !projection_matrix) {
        // printf("Error: Memory allocation failed.\n");
        return -1;
    }

    // printf("Starting Vision Transformer input processing...\n");
    // printf("Image data starts at virtual address: 0x%" PRIx64 "\n", (uint64_t)img_data);
    // printf("Output data starts at virtual address: 0x%" PRIx64 "\n", (uint64_t)output_data);
    // printf("Projection matrix starts at virtual address: 0x%" PRIx64 "\n", (uint64_t)projection_matrix);

    // Initialize the projection matrix with random values
    // printf("Initializing projection matrix with random values...\n");
    srand(0); // Seed the random number generator

    for (int i = 0; i < PROJ_DIM; i++) {
        for (int j = 0; j < PATCH_DIM; j++) {
            // Assign random float values between -1 and 1
            projection_matrix[i * PATCH_DIM + j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    }
    // printf("Projection matrix initialized.\n");

    // Calculate the addresses for temporary data (placed after the projection matrix)
    uint64_t temp_data_addr = PROJ_MATRIX_ADDR + PROJ_MATRIX_SIZE;
    uint64_t temp_data_size = PATCH_DIM * sizeof(uint8_t) + PROJ_DIM * sizeof(float);

    // Verify temporary data does not exceed mapped range
    if (temp_data_addr + temp_data_size > 0x120000000ULL) {
        // printf("Error: Not enough memory for temporary data.\n");
        return -1;
    }

    // uint8_t *patch_flat = (uint8_t *)temp_data_addr; // Flattened patch
    // float *projected_vector = (float *)(temp_data_addr + PATCH_DIM * sizeof(uint8_t)); // Projected vector
    uint8_t *patch_flat = (uint8_t *)malloc(PATCH_DIM * sizeof(uint8_t));
    float *projected_vector = (float *)malloc(PROJ_DIM * sizeof(float));
    if (!patch_flat || !projected_vector) {
        // printf("Error: Memory allocation failed for temporary buffers.\n");
        return -1;
    }

    // printf("Temporary data starts at virtual address: 0x%" PRIx64 "\n", temp_data_addr);

    // Process each patch
    int num_patches_h = IMG_HEIGHT / PATCH_SIZE;  // Number of patches vertically (14)
    int num_patches_w = IMG_WIDTH / PATCH_SIZE;   // Number of patches horizontally (14)

    // printf("Processing patches...\n");
    for (int ph = 0; ph < num_patches_h; ph++) {
        for (int pw = 0; pw < num_patches_w; pw++) {
            // Calculate the starting address of the current patch in the image data
            uint8_t *patch_ptr = img_data +
                (ph * PATCH_SIZE * IMG_WIDTH * NUM_CHANNELS) +  // Row offset
                (pw * PATCH_SIZE * NUM_CHANNELS);               // Column offset

            // Print the starting address of the current patch
            // printf("Flattening patch at position (ph=%d, pw=%d), patch start address: 0x%" PRIx64 "\n",
                // ph, pw, (uint64_t)patch_ptr);

            // Flatten the patch into a vector
            int idx = 0;
            for (int i = 0; i < PATCH_SIZE; i++) {
                for (int j = 0; j < PATCH_SIZE; j++) {
                    for (int c = 0; c < NUM_CHANNELS; c++) {
                        patch_flat[idx++] = *(patch_ptr +
                            (i * IMG_WIDTH * NUM_CHANNELS) +  // Move down i rows
                            (j * NUM_CHANNELS) +              // Move right j columns
                            c);                               // Select the channel
                    }
                }
            }
            // printf("Patch flattened, starting linear projection...\n");

            // Perform linear projection: projected_vector = projection_matrix * patch_flat
            for (int i = 0; i < PROJ_DIM; i++) {
                projected_vector[i] = 0.0f;
                for (int j = 0; j < PATCH_DIM; j++) {
                    // Convert uint8_t to float for multiplication
                    projected_vector[i] += (float)patch_flat[j] * projection_matrix[i * PATCH_DIM + j];
                }
            }
            // printf("Linear projection completed for patch (ph=%d, pw=%d).\n", ph, pw);

            // Store the projected vector to the output data
            float *output_ptr = output_data + ((ph * num_patches_w + pw) * PROJ_DIM);
            // printf("Storing projected vector at address: 0x%" PRIx64 "\n", (uint64_t)output_ptr);

            for (int i = 0; i < PROJ_DIM; i++) {
                output_ptr[i] = projected_vector[i];
            }
            // printf("Projected vector stored for patch (ph=%d, pw=%d).\n", ph, pw);
        }
    }
    
    free(patch_flat);
    free(projected_vector);
    free(output_data);
    free(projection_matrix);
    
    // printf("Vision Transformer input processing completed.\n");

    return 0;
}
