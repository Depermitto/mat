#include <iostream>
#include <numeric>
#include <vector>

#include "mat.hpp"

int main() {
    std::vector<int> vec(500);
    std::iota(vec.begin(), vec.end(), 10);

    auto a = mat::Mat<6, 8>(vec);
    std::cout << a.mod_col<5>(!mat::vec<6>(0)).mod_row<3>(mat::vec<8>(0)) << "\n";
}
