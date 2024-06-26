![GitHub-Mark-Light](other/raptor_text_logo_small.png#gh-dark-mode-only)
![GitHub-Mark-Dark](other/raptor_text_logo_black_small.png#gh-light-mode-only)

# Raptor

## Description

- A high-level algorithmic skeleton C++ template library designed to ease the development of parallel programs using CUDA.
- Raptor's syntax and core design features are taken from [marrow](https://docentes.fct.unl.pt/p161/software/marrow-skeleton-framework). Simillarly to marrow, raptor provides a set of smart containers (`vector`, `array`, `vector<array>`, `scalar`) and skeletons (`map`, `reduce`, `scan`, `filter`) that can be applied over the smart containers. Containers can store the data both on the host and device, and expose a seamingly unified memory address space. Skeletons are executed on the device, and the containers are automatically and lazily allocated (if not already allocated) and uploaded to the device whenever a skeleton is executed. Similarly to marrow, raptor also provides a generic function primitive. A raptor function allows one to specify a generic device function that can operate over multiple raptor containers of different sizes, primitive data types, and the GPU coordinates.
- Read more about raptor in my [blog post](https://dma-neves.github.io/dma/raptor.html).

## Requirements

* [Nvidia Modern GPU](https://developer.nvidia.com/cuda-gpus) (compute capability &ge; 6.0).
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (only tested with v12).
* Host C++ compiler with support for C++14.
* [CMake](https://cmake.org) (v3.24 or greater).
* Unix based OS.

## Build and Run

```bash
git clone --recursive git@github.com:dma-neves/raptor.git
cd raptor
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
examples/riemann_sum 0 10 1000000
```

## Example: Riemann Sum

```c++
__device__
static float fun(float x) { ... }

struct compute_area {
    __device__
    float operator() (float index, int start, float dx) {

        float x = static_cast<float>(start) + index*dx;
        float y = fun(x);
        return dx * y;
    }
};


float riemann_sum(int start, int end, int samples) {

    float dx = static_cast<float>(end - start) / static_cast<float>(samples);
    raptor::vector<float> indexes = raptor::iota<float>(samples);
    raptor::vector<float> vals = raptor::map<compute_area>(indexes,start, dx);
    raptor::scalar<float> result = raptor::reduce<sum<float>>(vals);
    return result.get();
}

int main() {
    float rs = riemann_sum(0, 10, 1000000);
}
```

## Example: Montecarlo

```c++
struct montecarlo_fun : raptor::function<montecarlo_fun, out<float*>> {

    __device__
    void operator()(coordinates_t tid, float* result) {

        float x = raptor::random::rand(tid);
        float y = raptor::random::rand(tid);

        result[tid] = (x * x + y * y) < 1;
    }
};

float pi_montecarlo_estimation(int size) {

    montecarlo_fun montecarlo;
    // montecarl.set_size(size); /*optional*/
    raptor::vector<float> mc_results(size);
    montecarlo.apply(mc_results);
    raptor::scalar<float> pi = raptor::reduce<sum<float>>(mc_results);
    return pi.get() / (float)size * 4.f;
}

int main() {

    float pi = pi_montecarlo_estimation(1000000);
}
```
