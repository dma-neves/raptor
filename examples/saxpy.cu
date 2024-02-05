#include "lmarrow/lmarrow.hpp"

using namespace lmarrow;

struct saxpy {

    __device__
    float operator()(float x, float y, float a) {

        return a*x + y;
    }
};


int main() {

	int n = 10;

    float a = 2.0f;
    vector<float> x(n);
    vector<float> y(n);

    x.fill_on_device(counting_sequence_filler<int>());
    y.fill_on_device(counting_sequence_filler<int>());

    vector<float> res = lmarrow::map<saxpy>(x,y,a);

    std::cout << "result  : ";
    for(int i = 0; i < n; i++) { std::cout << res[i] << " "; }
    std::cout << std::endl;

    std::cout << "expected: ";
    for(int i = 0; i < n; i++) { std::cout << ( (float)i * a + (float)i ) << " "; }
    std::cout << std::endl;
}