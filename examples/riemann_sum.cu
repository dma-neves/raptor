//
// Created by david on 05-01-2024.
//

#include "lmarrow.hpp"

using namespace lmarrow;

__device__
static float fun(float x) {

    return pow(x, 4.f) * (-5.56) + pow(x, 3.f) * 1.34 + x * x *  3.45 + x * 5 + 40;
}

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

int main(int argc, char *argv[]) {

    if(argc != 4) {

        std::cout << "Usage: " << argv[0] << " <start> <end> <samples>" << std::endl;
        return 0;
    }


    int start = (std::size_t)std::stoi(argv[1]);
    int end = (std::size_t)std::stoi(argv[2]);
    int samples = (std::size_t)std::stoi(argv[3]);

    float rs = riemann_sum(start, end, samples);

    std::cout << "riemann sum: " << rs << std::endl;
}

