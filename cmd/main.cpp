#include <iostream>
#include <numeric>
#include <vector>

#include "mat.hpp"

int main() {
    std::vector<int> vec(50);
    std::iota(vec.begin(), vec.end(), 10);

    auto b = mat::square<4>(vec);
    std::cout << b << "\n";

    std::cout << b - mat::identity<4>().mul(20) << "\n";
}
