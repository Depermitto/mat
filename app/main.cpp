#include <iostream>
#include <numeric>

#include "mamut.hpp"

int main() {
    auto a = mat::Mamut<int, 3, 4>();
    std::iota(a.begin(), a.end(), 10);
    std::cout << a << "\n";
}
