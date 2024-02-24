#include <iostream>

#include "mat.hpp"

int main() {
    auto a = mat::vec2<double>(2, 3);
    std::cout << 20 * a << "\n";
    
    auto b = mat::Mat<2, 6>({1, 2, 3, 4});
    std::cout << a * b << "\n";
    
    std::cout << (a != b) << "\n";
}