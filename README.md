![GitHub-Mark-Light](other/raptor_text_logo_small.png#gh-dark-mode-only)
![GitHub-Mark-Dark](other/raptor_text_logo_black_small.png#gh-light-mode-only)

# Raptor

## Description

- A high-level algorithmic skeleton C++ template library designed to ease the development of parallel programs using CUDA.
- Raptor's syntax and core design features are taken from [marrow](https://docentes.fct.unl.pt/p161/software/marrow-skeleton-framework). Simillarly to marrow, raptor provides a set of smart containers (`vector`, `array`, `vector<array>`, `scalar`) and skeletons (`map`, `reduce`, `scan`, `filter`) that can be applied over the smart containers. Containers can store the data both on the host and device, and expose a seamingly unified memory address space. Skeletons are executed on the device, and the ontainers are automatically and lazily allocated (if not already allocated) and uploaded to the device whenever a skeleton is executed. Similarly to marrow, raptor also provides a function primitive, which allows one to specify a generic device function that operates over multiple raptor containers of different sizes, primitive data types, and the GPU coordinates.
## Motivation

- Raptor was developed as a simplified and lighter weight alternative to marrow (when using a CUDA backend). For complex irregular applications (many data-dependencies; potential for communication/computation overlap; nested operations over containers), marrow can acheive better performance. For more regular and bulk-synchronous applications, raptor can take advantage of a minimal runtime.
- You may note that there already exists a standard high-level parallel algorithms library that tries to achieve some of the same goals as raptor: [thrust](https://developer.nvidia.com/thrust). The main differentiating features of raptor are:
    - The adoption of a unified address space, where containers ensure the necessary synchronization automatically in a lazy manner.
    - Multiple container types (`vector`, `array`, `vector<array>`, `scalar`).
    - Ability to specify a synchronization granularity (coarse grain - whole container is lazily synchronized, fine grain - only dirty_host elements are lazily synchronized). 
    - Powerfull generic function primitive.

## Todo
- Add lazy copies and fills
- Make vector\<array\> copies and fills async like in vector_base
- Try using streams when uploading skeleton argument containers
- Add more tests and examples (`vector<array>`, `function`, `util`)

## Requirements

* [Nvidia Modern GPU](https://developer.nvidia.com/cuda-gpus) (compute capability &ge; 6.0).
* [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (only tested with v12).
* Host C++ compiler with support for C++14.
* [CMake](https://cmake.org) (v3.24 or greater).
* Unix based OS.

## Build and Run

```bash
mkdir build
cd build
cmake ..
make
examples/riemann_sum 0 10 1000000
```

## Riemann Sum Example

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
    vector<float> indexes = iota<float>(samples);
    vector<float> vals = map<compute_area>(indexes,start, dx);
    scalar<float> result = reduce<sum<float>>(vals);
    return result.get();
}

int main() {
    float rs = riemann_sum(0, 10, 1000000);
}
```

## Mandelbrot Example

```c++
__device__
int inline divergence(int depth, raptor::math::complex<float> c0) {

    raptor::math::complex<float> c = c0;
    int i = 0;
    while (i < depth && c.dot() < TOL) {
        c = c0 + (c * c);
        i++;
    }
    return i;
}

struct mandelbrot_fun {

    static constexpr float center_x = -1.5f;
    static constexpr float center_y = -1.5f;
    static constexpr float scale_x = 3.f;
    static constexpr float scale_y = 3.f;

    __device__
    int operator()(int index, int width, int height) const {

        float x = (float)(index % height);
        float y = (float)(index / height);

        raptor::math::complex<float> c0(center_x + (x / (float)width) * scale_x ,
                                         center_y + (y / (float)height) * scale_y);

        return divergence(DEPTH, c0);
    }
};

vector<int> compute_mandelbrot(int n) {

    vector<int> indexes = iota<int>(n*n);
    vector<int> result = map<mandelbrot_fun>(indexes, n, n);
    return result;
}

int main() {

    vector<int> mandelbrot = compute_mandelbrot(1000);
    render(mandelbrot, DEPTH);
}
```