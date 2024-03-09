#include <iostream>
#include <numeric>

#include "mamut.hpp"

int main() {
    auto a = mat::Mamut<int, 4, 5>();
    auto b = mat::Mamut<int, 4, 5>(1, 2, 3, 4, 5);

    a.add(b).mod_col<2>(0, 7, 8, 6);
    std::iota(a.begin(), a.end(), 0);

    std::cout << a << "\n";
    std::cout << b << "\n";
}
