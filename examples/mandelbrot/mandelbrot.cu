//
// Created by david on 02-01-2024.
//

#include "raptor.hpp"

#define DEPTH 1000
#define TOL 4.0f

#include "mandelbrot_render.hpp"

using namespace raptor;

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

    static constexpr float center_x = -2.f;
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

int main(int argc, char **argv) {


    if(argc != 2) {

        std::cout << "Usage: " << argv[0] << " <width>" << std::endl;
        return 0;
    }

    int n;
    n = atoi(argv[1]);

    vector<int> mandelbrot = compute_mandelbrot(n);
    render(mandelbrot, DEPTH);
}