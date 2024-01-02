#include "lmarrow/containers/vector.hpp"
#include "lmarrow/function.hpp"
#include "lmarrow/skeletons/map.hpp"

using namespace lmarrow;

struct sequence {

    __device__
    float operator()(std::size_t i) {
        return (float)i;
    }
};

struct saxpy {

    __device__
    float operator()(float x, float y, float a) {

        return a*x + y;
    }
};


int main() {

	sequence seq;
	int n = 10;

    float a = 2.0f;
    vector<float> x(n);
    vector<float> y(n);

    x.fill_on_device(seq);
    y.fill_on_device(seq);

    vector<float> res = lmarrow::map<saxpy>(x,y,a);

    std::cout << "result  : ";
    for(int i = 0; i < n; i++) {

        std::cout << res[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "expected: ";
    for(int i = 0; i < n; i++) {

        float expected = (float)i * a + (float)i;
        std::cout << expected << " ";
    }
    std::cout << std::endl;
}