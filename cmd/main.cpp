#include <iostream>

#include "mamut.hpp"

int main() {
    auto v = std::vector<int>(50);
    std::iota(v.begin(), v.end(), 0);

    auto b = mat::Mamut<int, 2, 7>(v.begin(), 14);
    auto c = mat::Mamut<int, 7, 2>(v.begin() + 14, 14);
    b += !c;
    std::cout << b << "\n";

    auto id = mat::Mamut<int, 5, 5>::identity();
    std::cout << id << "\n";
}
