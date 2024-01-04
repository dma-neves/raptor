# lmarrow

## Description

- A high-level C++ template library designed to ease the development of parallel programs using CUDA.
- lmarrow's syntax and core design are taken from marrow. Simillarly to marrow, lmarrow provides a set of smart containers (vector, array, vector\<array\> and scalar) and skeletons (map, reduce, scan, filter) that can be applied over the smart containers. The containers can store data both on the GPU and CPU, and synchronize automatically whenever necessary.
- marrow provides a function primitive that behaves as a more flexible map skeleton, allowing one to handle multiple containers with different sizes. lmarrow also provides a function primitive, but a less refined one which is treated independantly of the map skeleton.
- Contrary to marrow, lmarrow doesn't allow the composition of skeletons into a single kernel, nor does it automatically track dependencies and use streams to parallelize operations that could be performed asynchronously. In lmarrow everything is performed in a bulk-synchronous manner.
- lmarrow was developed as a simplified and lighter weight alternative to marrow. For complex applications with many data-dependencies, potential for communication/computation overlap and complex operations over containers, marrow will most likely have better performance. For simpler or more bulk-synchronous-oriented applications, lmarrow might be enough and it can take advantage of less runtime overheads.
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

    sequence_fill seq;

    float a = 2.0f;
    int n = 10;
    vector<float> x(n);
    vector<float> y(n);

    x.fill_on_device(seq);
    y.fill_on_device(seq);

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
    mc_results.dirty(); // lmarrow can't automatically detect container updates on the device

    scalar<float> pi = reduce<sum<float>>(mc_results);

    return pi.get_data() / (float)size * 4.f;
}
```
