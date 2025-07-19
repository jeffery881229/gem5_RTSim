#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h> // For rand()
#include <time.h>   // For seeding rand()

#define IMG_HEIGHT 224
#define IMG_WIDTH 224
#define PATCH_SIZE 32
#define NUM_CHANNELS 3
#define NUM_PATCHES ((IMG_HEIGHT / PATCH_SIZE) * (IMG_WIDTH / PATCH_SIZE)) // 196
#define PATCH_DIM (PATCH_SIZE * PATCH_SIZE * NUM_CHANNELS) // 16 x 16 x 3 = 768
#define PROJ_DIM 384  // Projection dimension (adjusted as needed)
#define INPUT_ADDR 0x100000000ULL      // Starting virtual address of the image
#define PIM_ADDR 0xf00000000ULL      // Starting virtual address of the PIM
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
        return -1;
    }

    // Pointers to the memory-mapped regions
    uint8_t *img_data = (uint8_t *)INPUT_ADDR;         // Input image data
    uint8_t *access_pim_addr = (uint8_t *)PIM_ADDR;        // Access PIM address

    // try to access wrong virtual address
    uint8_t *wrong_vir_addr = (uint8_t *)malloc(sizeof(uint8_t));
    wrong_vir_addr[0] = *access_pim_addr;

    // Total number of data chunks to read
    uint64_t total_chunks = PROJ_DIM * NUM_PATCHES;

    // Variable to store the fetched data
    uint16_t data_value;

    // Loop to read the data chunks
    for (uint64_t i = 0; i < total_chunks; i++) {
        // Calculate the address to read from
        uint64_t addr = INPUT_ADDR + i * sizeof(uint16_t);

        // Access the data at the calculated address
        data_value = *((uint16_t *)addr);

        // Since we can overwrite, we can reuse data_value
        // Optionally, you can process the data_value here
        // For example, print the data:
        // printf("Data chunk %llu: 0x%04x\n", i, data_value);
    }

    return 0;
    

    free(wrong_vir_addr);
    

    return 0;
}
