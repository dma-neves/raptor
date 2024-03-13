//
// Created by david on 13-10-2023.
//

#include <gtest/gtest.h>
#include <chrono> // Added for timing

#include "lmarrow/lmarrow.hpp"

using namespace lmarrow;

#define block_size_16 16 / sizeof(int)
#define block_size_32 32 / sizeof(int)
#define block_size_64 64 / sizeof(int)
#define block_size_128 128 / sizeof(int)


TEST(VectorOfArrayBlockSize, BlockSizes) {

    cudaFree(0);

    // ########################## 16 Bytes ##########################

    vector<array<int, block_size_16>> vec16(1024);
    vec16.fill(1);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    vec16.upload();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "16 Byte Block Upload Time: " << duration.count() << " microseconds" << std::endl;

    // ########################## 32 Bytes ##########################

    vector<array<int, block_size_32>> vec32(1024);
    vec32.fill(1);
    cudaDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    vec32.upload();
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "32 Byte Block Upload Time: " << duration.count() << " microseconds" << std::endl;

    // ########################## 64 Bytes ##########################

    vector<array<int, block_size_64>> vec64(1024);
    vec64.fill(1);
    cudaDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    vec64.upload();
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "64 Byte Block Upload Time: " << duration.count() << " microseconds" << std::endl;

    // ########################## 128 Bytes ##########################

    vector<array<int, block_size_128>> vec128(1024);
    vec128.fill(1);
    cudaDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    vec128.upload();
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "128 Byte Block Upload Time: " << duration.count() << " microseconds" << std::endl;
}
