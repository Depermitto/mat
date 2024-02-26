#include <iostream>
#include <numeric>
#include <vector>

#include "mat.hpp"

int main() {
    std::vector<int> vec(500);
    std::iota(vec.begin(), vec.end(), 10);

    auto a = mat::Mat<3, 2>(vec);
    std::cout << a << "\n";

    std::cout << a.mod_row<1>(mat::vec2(0, 1)) << "\n";
    std::cout << a.mod_col<0>(!mat::vec<3>(200)) << "\n";
}
