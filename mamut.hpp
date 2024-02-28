#pragma once

#include <cmath>
#include <format>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ranges>

namespace mat {

template <typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, bool> = true>
concept TrivialNumber = std::convertible_to<T, double>;

template <uint Rows, uint Cols>
concept Vector = (Rows == 1 or Cols == 1);

template <TrivialNumber T, uint Rows, uint Cols>
class Mamut {
    using array = T[Rows * Cols];
    using iter = T*;
    using const_iter = const T*;

   protected:
    array data{};

    auto arithmetic_operation(const Mamut& other, std::function<T(T, T)>&& f) -> Mamut& {
        for (auto [i, j] : std::views::zip(*this, other)) {
            i = f(i, j);
        }
        return *this;
    }

   public:
    Mamut() = default;
    explicit Mamut(TrivialNumber auto... data) : data(data...) {}

    Mamut(std::forward_iterator auto begin, std::forward_iterator auto end) {
        std::copy(begin, end, this->begin());
    }
    Mamut(std::forward_iterator auto begin, size_t n) {
        std::copy_n(begin, n, this->begin());
    }

    static auto identity() -> Mamut
        requires(Rows == Cols)
    {
        auto mamut = Mamut<T, Rows, Cols>();
        for (uint i = 0; i < Rows; i++) {
            mamut[i, i] = 1;
        }
        return mamut;
    }

    auto begin() -> iter {
        return std::begin(data);
    }
    auto begin() const -> const_iter {
        return std::cbegin(data);
    }

    auto end() -> iter {
        return std::end(data);
    }
    auto end() const -> const_iter {
        return std::cend(data);
    }

    constexpr auto extends() const -> std::pair<uint, uint> {
        return std::make_pair(Rows, Cols);
    }
    constexpr auto size() const -> uint {
        return Rows * Cols;
    }

    auto get(uint i) const -> std::optional<T> {
        if (i >= size()) {
            return std::nullopt;
        }
        return *(begin() + i);
    }
    auto get(uint i, uint j) const -> std::optional<T> {
        auto const [_, c] = extends();
        return get(i * c + j);
    }

    auto at(uint i, uint j) -> T& {
        auto const [r, c] = extends();
        if (i >= r or j >= c) {
            throw std::out_of_range(
                std::format("array index [{}, {}] out of index bounds [{}, {}]", i, j, r - 1, c - 1));
        }
        return *(begin() + i * c + j);
    }
    auto at(uint i) -> T& {
        if (i >= size()) {
            throw std::out_of_range(std::format("array index [{}] out of index bound [{}]", i, size()));
        }
        return *(begin() + i);
    }

    auto operator[](uint i) -> T& {
        return at(i);
    }
    auto operator[](uint i, uint j) -> T& {
        return at(i, j);
    }

    friend auto operator<<(std::ostream& os, const Mamut& mat) -> std::ostream& {
        auto parsed = mat | std::views::transform([](auto i) { return std::formatted_size("{}", i); });
        int width = *std::max_element(parsed.begin(), parsed.end());
        auto [rows, cols] = mat.extends();

        os << "[";
        if (rows != 1) {
            os << "[";
        }

        for (uint i = 0; i != rows; i++) {
            for (uint j = 0; j != cols; j++) {
                os << std::left << std::setw(width) << mat.get(i, j).value();
                if (j + 1 < cols) {
                    os << " ";
                }
            }
            os << "]";
            if (i + 1 < rows) {
                os << "\n [";
            }
        }

        if (rows != 1) {
            os << "]";
        }
        return os;
    }

    auto operator==(const Mamut& other) const -> bool {
        return std::equal(this->begin(), this->end(), other.begin());
    }
    auto operator!=(const Mamut& other) const -> bool {
        return not(*this == other);
    }

    auto clone() const -> Mamut {
        return *this;
    }

    template <uint R>
    auto mul(const Mamut<T, Cols, R>& other) const -> Mamut<T, Rows, R> {
        Mamut<T, Rows, R> mat{};
        for (uint i = 0; i < Rows; ++i) {
            for (uint k = 0; k < R; ++k) {
                for (uint n = 0; n < Cols; ++n) {
                    mat[i, k] = this->get(i, n).value() * other.get(n, k).value();
                }
            }
        }
        return mat;
    }
    auto mul(TrivialNumber auto constant) -> Mamut {
        std::for_each(begin(), end(), [constant](T& i) { i *= constant; });
        return *this;
    }

    auto add(const Mamut& other) -> Mamut& {
        return arithmetic_operation(other, std::plus<T>());
    }
    auto operator+(const Mamut& other) -> Mamut& {
        return this->clone().add(other);
    }
    auto operator+=(const Mamut& other) -> Mamut& {
        return this->add(other);
    }

    auto sub(const Mamut& other) -> Mamut& {
        return arithmetic_operation(other, std::minus<T>());
    }

    auto hadamard(const Mamut& other) -> Mamut& {
        return arithmetic_operation(other, std::multiplies<T>());
    }

    auto transpose() const -> Mamut<T, Cols, Rows> {
        auto transposed = Mamut<T, Cols, Rows>();
        for (uint i = 0; i < Rows; ++i) {
            for (uint j = 0; j < Cols; ++j) {
                transposed[j, i] = this->get(i, j).value();
            }
        }
        return transposed;
    }
    auto operator!() const -> Mamut<T, Cols, Rows> {
        return this->transpose();
    }

    template <uint R, uint C>
        requires((R <= Rows and C <= Cols) and not(R == Rows and C == Cols))
    auto cut() const -> Mamut<T, R, C> {
        return Mamut<T, R, C>(begin(), begin() + R * C);
    }

    template <uint R, uint C>
        requires((R >= Rows and C >= Cols) and not(R == Rows and C == Cols))
    auto pad() const -> Mamut<T, R, C> {
        return Mamut<T, R, C>(begin(), end());
    }

    template <uint R, uint C>
    auto resize_to() const -> Mamut<T, R, C> {
        return Mamut<T, R, C>(begin(), begin() + std::min(R * C, Rows * Cols));
    }

    template <uint R>
        requires(R < Rows)
    auto row() const -> Mamut<T, 1, Cols> {
        return Mamut<T, 1, Cols>(begin() + R * Cols, Cols);
    }

    template <uint R>
        requires(R < Rows)
    auto mod_row(auto&&... args) -> Mamut& {
        auto row = Mamut<T, 1, Cols>(args...);
        for (uint i = 0; i < Cols; ++i) {
            this->at(R, i) = row[0, i];
        }
        return *this;
    }

    template <uint C>
        requires(C < Cols)
    auto col() const -> Mamut<T, Rows, 1> {
        Mamut<T, Rows, 1> col{};
        for (uint j = 0; j < Rows; ++j) {
            col[j, 0] = this->get(j, C).value();
        }
        return col;
    }

    template <uint C>
        requires(C < Cols)
    auto mod_col(auto&&... args) -> Mamut& {
        auto col = Mamut<T, Rows, 1>(args...);
        for (uint j = 0; j < Rows; ++j) {
            this->at(j, C) = col[j, 0];
        }
        return *this;
    }

    // Vectors
    auto mag_squared() const -> T
        requires Vector<Rows, Cols>
    {
        return std::accumulate(begin(), end(), T(), [](auto sum, auto i) { return sum + std::pow(i, 2); });
    }

    auto mag() const -> double
        requires Vector<Rows, Cols>
    {
        return std::sqrt(mag_squared());
    }

    auto norm() const -> std::optional<Mamut<double, Rows, Cols>>
        requires Vector<Rows, Cols>
    {
        auto mag = this->mag();
        if (mag == 0) {
            return std::nullopt;
        }

        auto clone = Mamut<double, Rows, Cols>(begin(), end());
        std::for_each(clone.begin(), clone.end(), [mag](auto& i) { i /= mag; });
        return clone;
    }

    auto direction(const Mamut& other) const -> std::optional<Mamut<double, Rows, Cols>> {
        return other.sub(*this).norm();
    }

    template <uint R, uint C>
    auto dot(const Mamut<T, R, C>& other) const -> T
        requires Vector<Rows, Cols> and Vector<R, C> and (R * C == Rows * Cols)
    {
        T sum{};
        for (uint i = 0; i < Rows; ++i) {
            for (uint j = 0; j < Cols; ++j) {
                sum += this->get(std::min(i, j), std::max(i, j)).value() * other.get(i, j).value();
            }
        }
        return sum;
    }
};
}  // namespace mat