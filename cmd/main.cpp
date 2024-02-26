#include <iostream>
#include <numeric>
#include <vector>

#include "mat.hpp"

int main() {
    std::vector<int> vec(500);
    std::iota(vec.begin(), vec.end(), 10);

    auto a = mat::Mat<6, 8>(vec);
    std::cout << a << "\n";
    
    a = a & mat::square<4>(1).pad<6, 8>();
    std::cout << a << "\n";
}
