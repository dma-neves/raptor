#include "raptor.hpp"

using namespace raptor;

struct saxpy {

    __device__
    float operator()(float x, float y, float a) {

        return a*x + y;
    }
};


int main() {

	int n = 10;

    float a = 2.0f;

    vector<float> x = iota<float>(n);
    vector<float> y = iota<float>(n);

    vector<float> res = raptor::map<saxpy>(x,y,a);

    std::cout << "result  : ";
    for(int i = 0; i < n; i++) { std::cout << res[i] << " "; }
    std::cout << std::endl;

    std::cout << "expected: ";
    for(int i = 0; i < n; i++) { std::cout << ( (float)i * a + (float)i ) << " "; }
    std::cout << std::endl;
}