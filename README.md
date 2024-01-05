# lmarrow

## Description

- A high-level C++ template library designed to ease the development of parallel programs using CUDA.
- lmarrow's syntax and core design are taken from [marrow](https://docentes.fct.unl.pt/p161/software/marrow-skeleton-framework). Simillarly to marrow, lmarrow provides a set of smart containers (vector, array, vector\<array\> and scalar) and skeletons (map, reduce, scan, filter) that can be applied over the smart containers. Containers can store the data both on the GPU and CPU. Skeletons are executed on the GPU. The containers are automatically and lazily allocated and uploaded to the GPU whenever a skeleton is executed, and lazily synchronized to the host, whenever a host access or update is performed.
- marrow provides a function primitive that behaves as a more flexible map skeleton, allowing one to handle multiple containers with different sizes. lmarrow also provides a function primitive, but a less refined one which is treated independantly of the map skeleton.
- A lmarrow-function, allows one to specify a generic device function that operates over lmarrow containers. primitive data types, and the GPU coordinates (thread ID).
- Contrary to marrow, lmarrow doesn't allow the nesting of multiple skeletons into a single kernel, nor does it automatically track data dependencies and use streams and events to parallelize operations that could be performed asynchronously. In lmarrow, by default, everything is performed using the default stream (stream 0). Non default-streams can be used (not thoroughly tested), but all the synchronization must be ensured by the programmer.
- lmarrow was developed as a simplified and lighter weight alternative to marrow. For complex applications with many data-dependencies, potential for communication/computation overlap and complex operations over containers, marrow will most likely have better performance. For simpler or more bulk-synchronous-oriented applications, lmarrow might be enough and can take advantage of less runtime overheads.
- Internally, besides the base CUDA primitives, lmarrow utilizes the CUB and thrust libraries.

## Saxpy Example

```c++
struct saxpy {

    __device__
    float operator()(float x, float y, float a) {

        return a*x + y;
    }
};

int main() {
    
    float a = 2.0f;
    int n = 10;
    vector<float> x(n);
    vector<float> y(n);
    x.fill_on_device(counting_sequence_filler<int>());
    y.fill_on_device(counting_sequence_filler<int>());

    vector<float> saxpy_res = lmarrow::map<saxpy>(x,y,a);
}
```

## Montecarlo Example

```c++
struct montecarlo_fun : lmarrow::function_with_coordinates<montecarlo_fun> {

    __device__
    int operator()(coordinates_t tid, float* result) {

        float x = lmarrow::random::random(tid);
        float y = lmarrow::random::random(tid);

        result[tid] = (x * x + y * y) < 1;
    }
};

float pi_montecarlo_estimation(int size) {

    montecarlo_fun montecarlo;

    vector<float> mc_results(size);
    montecarlo.apply(size, mc_results);
    mc_results.dirty_on_device(); // lmarrow can't automatically detect container updates on the device

    scalar<float> pi = reduce<sum<float>>(mc_results);

    return pi.get_data() / (float)size * 4.f;
}
```
