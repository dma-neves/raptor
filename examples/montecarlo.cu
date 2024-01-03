//
// Created by david on 02-01-2024.
//

#include "lmarrow/lmarrow.hpp"

using namespace lmarrow;

struct montecarlo_fun : function_with_coordinates<montecarlo_fun> {

    __device__
    int operator()(coordinates_t tid, float* result) {

        float x = lmarrow::random::rand(tid);
        float y = lmarrow::random::rand(tid);

        result[tid] = (x * x + y * y) < 1;
    }
};

float pi_montecarlo_estimation(int size) {

    montecarlo_fun montecarlo;

    vector<float> mc_results(size);
    montecarlo.apply(size, mc_results);

    scalar<float> pi = reduce<sum<float>>(mc_results);

    return pi.get_data() / (float)size * 4.f;
}

int main(int argc, char *argv[]) {


    if(argc != 2) {

        std::cout << "Usage: " << argv[0] << " <nelementes>" << std::endl;
        return 0;
    }

    std::size_t size = (std::size_t)std::stoi(argv[1]);

    float pi = pi_montecarlo_estimation(size);
    std::cout << "pi: " << pi << std::endl;
}