# lmarrow

## Description

- A high-level algorithmic skeleton C++ template library designed to ease the development of parallel programs using CUDA.
- lmarrow offers a subset of [marrow](https://docentes.fct.unl.pt/p161/software/marrow-skeleton-framework)'s syntax and core features. Simillarly to marrow, lmarrow provides a set of smart containers (`vector`, `array`, `vector<array>`, `scalar`) and skeletons (`map`, `reduce`, `scan`, `filter`) that can be applied over the smart containers. Containers can store the data both on the GPU and CPU, and expose a seamingly unified memory address space. Skeletons are executed on the GPU. The containers are automatically and lazily allocated (if not already allocated) and uploaded to the GPU whenever a skeleton is executed. Similarly to marrow, lmarrow also provides a function primitive, which allows one to specify a generic device function that operates over multiple lmarrow containers of different sizes, primitive data types, and the GPU coordinates (thread ID).
- Contrary to marrow, lmarrow doesn't support multiple backends, doesn't allow the nesting of multiple skeletons into a single kernel, nor does it automatically track data dependencies and use streams and events to parallelize operations that could be performed asynchronously. In lmarrow, by default, most operations are performed on the default stream (stream 0).

## Requirements

* [Nvidia Modern GPU](https://developer.nvidia.com/cuda-gpus) (compute capability &ge; 6.0).
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (only tested with v12).
* Host C++ compiler with support for C++14.
* [CMake](https://cmake.org) (v3.24 or greater).
* Unix based OS.

## Build and Run

```bash
git clone --recurse-submodules git@github.com:dma-neves/lmarrow.git
cd lmarrow
mkdir build
cd build
cmake ..
make
examples/riemann_sum 0 10 1000000
```
