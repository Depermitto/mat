#include <iostream>

#include "mat.hpp"

int main() {
    auto b = mat::Mat<4, 6>({1, 2, 3, 4});
    std::cout << b << "\n" << b.submatrix<1, 3>(0, 2).value_or(mat::vec<3>()) << "\n";
}
