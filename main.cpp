#include <iostream>

#include "mat.hpp"

int main() {
    auto id = mat::Identity<5>();
    std::cout << id << "\n";

    auto a = mat::Mat<2, 5>({1, 2, 3, 4, 5, 6});
    std::cout << a * 2 << "\n";
    
    std::cout << a * id << "\n";
    
    auto b = mat::Vec3(1, 2, 3);
    std::cout << b << "\n" << b * 2 << "\n";
}