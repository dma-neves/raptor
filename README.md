# lmarrow

## Description

- A high-level C++ template library designed to ease the development of parallel programs using CUDA.
- lmarrow's syntax and core design are taken from [marrow](https://docentes.fct.unl.pt/p161/software/marrow-skeleton-framework). Simillarly to marrow, lmarrow provides a set of smart containers (vector, array, vector\<array\> and scalar) and skeletons (map, reduce, scan, filter) that can be applied over the smart containers. Containers can store the data both on the GPU and CPU, and expose a seamingly unified memory address space. Skeletons are executed on the GPU. The containers are automatically and lazily allocated and uploaded to the GPU whenever a skeleton is executed, and lazily synchronized to the host, whenever a host access or update is performed.
- marrow provides a function primitive that behaves as a more flexible map skeleton, allowing one to handle multiple containers with different sizes. lmarrow also provides a function primitive, but a less refined one which is treated independantly of the map skeleton.
- A lmarrow-function, allows one to specify a generic device function that operates over lmarrow containers. primitive data types, and the GPU coordinates (thread ID).
- Contrary to marrow, lmarrow doesn't allow the nesting of multiple skeletons into a single kernel, nor does it automatically track data dependencies and use streams and events to parallelize operations that could be performed asynchronously. In lmarrow, by default, everything is performed using the default stream (stream 0). Non default-streams can be used (not thoroughly tested), but all the synchronization must be ensured by the programmer.
- lmarrow was developed as a simplified and lighter weight alternative to marrow. For complex applications with many data-dependencies, potential for communication/computation overlap and complex operations over containers, marrow will most likely have better performance. For simpler or more bulk-synchronous-oriented applications, lmarrow might be enough and can take advantage of less runtime overheads.

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
    vector<float> indexes(samples);
    indexes.fill_on_device(counting_sequence_filler<float>());
    vector<float> vals = map<compute_area>(indexes,start, dx);
    scalar<float> result = reduce<sum<float>>(vals);
    return result.get_data();
}

int main() {
    float rs = riemann_sum(0, 10, 1000000);
}
```

## Mandelbrot Example

```c++
__device__
int inline divergence(int depth, lmarrow::math::complex<float> c0) {

    lmarrow::math::complex<float> c = c0;
    int i = 0;
    while (i < depth && dot(c) < TOL) {
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

        lmarrow::math::complex<float> c0(center_x + (x / (float)width) * scale_x ,
                                         center_y + (y / (float)height) * scale_y);

        return divergence(DEPTH, c0);
    }
};

vector<int> compute_mandelbrot(int n) {

    vector<int> indexes(n*n);
    indexes.fill_on_device(counting_sequence_filler<int>());
    vector<int> result = map<mandelbrot_fun>(indexes, n, n);
    return result;
}

int main() {

    vector<int> mandelbrot = compute_mandelbrot(1000);
    render(mandelbrot, DEPTH);
}
```

![alt text](other/mandelbrot.png)