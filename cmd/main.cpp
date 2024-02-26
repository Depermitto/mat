#include <iostream>
#include <numeric>
#include <vector>

#include "mat.hpp"

int main() {
    std::vector<int> vec(500);
    std::iota(vec.begin(), vec.end(), 10);

    auto a = mat::Mat<30, 40>(vec).cut<2, 8>();
    std::cout << a << "\n";
}
