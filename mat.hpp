#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <iomanip>
#include <span>
#include <sstream>

/**
 * namespace mat is home for the matrix Mat class, as well as the common interfaces all the specializations
 * implement. A lot of methods don't have documentation, mainly because the code has been written using macros
 * and (as the time of writing 25.02.2024) I don't know how to document macro code yet.
 */
namespace mat {

namespace detail {
namespace matrix {
#define MUL(T, Rows, Cols)                                                                        \
    /**                                                                                           \
     * Performs standard matrix multiplication.                                                   \
     */                                                                                           \
    template <uint R>                                                                             \
    auto mul(const Interface<Cols, R, T> &other) const noexcept -> Mat<Rows, R, T> {              \
        Mat<Rows, R, T> mat{};                                                                    \
        for (uint i = 0; i < Rows; ++i) {                                                         \
            for (uint k = 0; k < R; ++k) {                                                        \
                for (uint n = 0; n < Cols; ++n) {                                                 \
                    mat[i, k] += this->get(i, n).value() * other.get(n, k).value();               \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
        return mat;                                                                               \
    }                                                                                             \
                                                                                                  \
    template <uint R>                                                                             \
    inline auto operator*(const Interface<Cols, R, T> &other) const noexcept -> Mat<Rows, R, T> { \
        return this->mul(other);                                                                  \
    }

#define MUL_CONSTANT(T, Rows, Cols)                                                             \
    /**                                                                                         \
     *  Performs multiplication of all elements in the matrix by a constant.                    \
     */                                                                                         \
    template <TrivialNumber Number>                                                             \
    auto mul(Number constant) const -> Mat<Rows, Cols, T> {                                     \
        Mat<Rows, Cols, T> mat{};                                                               \
        for (uint i = 0; i < Rows; ++i) {                                                       \
            for (uint j = 0; j < Cols; ++j) {                                                   \
                mat[i, j] = this->get(i, j).value() * constant;                                 \
            }                                                                                   \
        }                                                                                       \
        return mat;                                                                             \
    }                                                                                           \
                                                                                                \
    template <TrivialNumber Number>                                                             \
    friend auto operator*(const Mat<Rows, Cols, T> &mat, Number constant)->Mat<Rows, Cols, T> { \
        return mat.mul(constant);                                                               \
    }                                                                                           \
                                                                                                \
    template <TrivialNumber Number>                                                             \
    friend auto operator*(Number constant, const Mat<Rows, Cols, T> &mat)->Mat<Rows, Cols, T> { \
        return mat.mul(constant);                                                               \
    }                                                                                           \
                                                                                                \
    template <TrivialNumber Number>                                                             \
    auto operator*=(Number constant)->Mat<Rows, Cols, T> {                                      \
        *this = this->mul(constant);                                                            \
        return *this;                                                                           \
    }

#define TRANSPOSE(T, Rows, Cols)                                   \
    /**                                                            \
     * Copies the contents and transpose them.                     \
     */                                                            \
    auto transpose() const noexcept -> Mat<Cols, Rows, T> {        \
        Mat<Cols, Rows, T> mat{};                                  \
        for (uint i = 0; i < Rows; ++i) {                          \
            for (uint j = 0; j < Cols; ++j) {                      \
                mat[j, i] = this->get(i, j).value();               \
            }                                                      \
        }                                                          \
        return mat;                                                \
    }                                                              \
                                                                   \
    inline auto operator!() const noexcept -> Mat<Cols, Rows, T> { \
        return this->transpose();                                  \
    }

#define ARITHMETIC(T, Rows, Cols, arithmetic_op, func_name, cpp_op)                                     \
    /**                                                                                                 \
     * Adds an arithmetic operation. (mul, add, sub, bit operations...)                                 \
     */                                                                                                 \
    auto func_name(const Mat<Rows, Cols, T> &other) const noexcept -> Mat<Rows, Cols, T> {              \
        Mat mat{};                                                                                      \
        for (uint i = 0; i < Rows; ++i) {                                                               \
            for (uint j = 0; j < Cols; ++j) {                                                           \
                mat[i, j] = this->get(i, j).value() arithmetic_op other.get(i, j).value();              \
            }                                                                                           \
        }                                                                                               \
        return mat;                                                                                     \
    }                                                                                                   \
                                                                                                        \
    inline auto operator cpp_op(const Mat<Rows, Cols, T> &other) const noexcept -> Mat<Rows, Cols, T> { \
        return this->func_name(other);                                                                  \
    }                                                                                                   \
                                                                                                        \
    inline auto operator cpp_op##=(const Mat<Rows, Cols, T> &other) noexcept -> Mat<Rows, Cols, T> {    \
        *this = this->func_name(other);                                                                 \
        return *this;                                                                                   \
    }

#define CUT(T, Rows, Cols)                                                                    \
    /**                                                                                       \
     * Cut a submatrix from one of the corners.                                               \
     */                                                                                       \
    template <uint R, uint C>                                                                 \
        requires Shrinkable<Rows, Cols, R, C>                                                 \
    auto cut() const noexcept -> Mat<R, C, T> {                                               \
        Mat<R, C, T> shrunk{};                                                                \
        for (uint i = 0; i < R; ++i) {                                                        \
            for (uint j = 0; j < C; ++j) {                                                    \
                shrunk[i, j] = this->get(i, i).value();                                       \
            }                                                                                 \
        }                                                                                     \
        return shrunk;                                                                        \
    }                                                                                         \
                                                                                              \
    /**                                                                                       \
     * Cut submatrix from a specified location within the original matrix,                    \
     * starting point denoted by (i, j).                                                      \
     */                                                                                       \
    template <uint R, uint C>                                                                 \
        requires Shrinkable<Rows, Cols, R, C>                                                 \
    auto submatrix(uint row, uint col) const noexcept -> std::expected<Mat<R, C, T>, Error> { \
        Mat<R, C, T> shrunk{};                                                                \
        for (uint i = 0; i < R; ++i) {                                                        \
            for (uint j = 0; j < C; ++j) {                                                    \
                if (not this->get(i + row, j + col).has_value()) {                            \
                    return std::unexpected(Error::OutOfRange);                                \
                }                                                                             \
                shrunk[i, j] = this->get(i + row, j + col).value();                           \
            }                                                                                 \
        }                                                                                     \
        return shrunk;                                                                        \
    }

#define PAD(T, Rows, Cols)                                                                         \
    /**                                                                                            \
     * Extend the matrix by padding new elements with the default value.                           \
     */                                                                                            \
    template <uint R, uint C>                                                                      \
        requires Extendable<Rows, Cols, R, C>                                                      \
    auto pad() const noexcept -> Mat<R, C, T> {                                                    \
        Mat<R, C, T> mat{};                                                                        \
        for (uint i = 0; i < Rows; ++i) {                                                          \
            for (uint j = 0; j < Cols; ++j) {                                                      \
                mat[i, j] = this->get(i, j).value();                                               \
            }                                                                                      \
        }                                                                                          \
        return mat;                                                                                \
    }                                                                                              \
                                                                                                   \
    template <uint R, uint C>                                                                      \
        requires Extendable<Rows, Cols, R, C>                                                      \
    auto expand(uint place_i, uint place_j) const noexcept -> std::expected<Mat<R, C, T>, Error> { \
        Mat<R, C, T> mat{};                                                                        \
        for (uint i = 0; i < Rows; ++i) {                                                          \
            for (uint j = 0; j < Cols; ++j) {                                                      \
                if (not mat.get(place_i + i, place_j + j).has_value()) {                           \
                    return std::unexpected(Error::OutOfRange);                                     \
                }                                                                                  \
                mat[place_i + i, place_j + j] = this->get(i, j).value();                           \
            }                                                                                      \
        }                                                                                          \
        return mat;                                                                                \
    }

#define ROW_COL(T, Rows, Cols)                          \
    template <uint R>                                   \
        requires(R < Rows)                              \
    auto row() const noexcept -> Mat<1, Cols, T> {      \
        Mat<1, Cols, T> row{};                          \
        for (uint i = 0; i < Cols; ++i) {               \
            row[0, i] = this->get(R, i).value();        \
        }                                               \
        return row;                                     \
    }                                                   \
                                                        \
    template <uint R>                                   \
        requires(R < Rows)                              \
    auto mod_row(Mat<1, Cols, T> new_row) -> Mat {      \
        for (uint i = 0; i < Cols; ++i) {               \
            this->at(R, i) = new_row.get(0, i).value(); \
        }                                               \
        return *this;                                   \
    }                                                   \
                                                        \
    template <uint C>                                   \
        requires(C < Cols)                              \
    auto col() const noexcept -> Mat<Rows, 1, T> {      \
        Mat<Rows, 1, T> col{};                          \
        for (uint j = 0; j < Rows; ++j) {               \
            col[j, 0] = this->get(j, C).value();        \
        }                                               \
        return col;                                     \
    }                                                   \
                                                        \
    template <uint C>                                   \
        requires(C < Cols)                              \
    auto mod_col(Mat<Rows, 1, T> new_col) -> Mat {      \
        for (uint j = 0; j < Rows; ++j) {               \
            this->at(j, C) = new_col.get(j, 0).value(); \
        }                                               \
        return *this;                                   \
    }

}  // namespace matrix
namespace vector {
#define DOT(T, Rows, Cols)                                                                          \
    /**                                                                                             \
     * Standard dot operation between two matrices                                                  \
     */                                                                                             \
    auto dot(const Interface<Rows, Cols, T> &other) const noexcept -> T {                           \
        T sum{};                                                                                    \
        for (uint i = 0; i < Rows; ++i) {                                                           \
            for (uint j = 0; j < Cols; ++j) {                                                       \
                sum += this->get(std::min(i, j), std::max(i, j)).value() * other.get(i, j).value(); \
            }                                                                                       \
        }                                                                                           \
        return sum;                                                                                 \
    }                                                                                               \
                                                                                                    \
    template <uint R>                                                                               \
    inline auto operator^(const Interface<Cols, R, T> &other) const noexcept -> T {                 \
        return this->dot(other);                                                                    \
    }

#define MAG(T, Rows, Cols)                                 \
    [[nodiscard]] auto mag_squared() const noexcept -> T { \
        T sum{};                                           \
        for (uint i = 0; i < Rows; i++) {                  \
            sum += std::pow(this->get(0, i).value(), 2);   \
        }                                                  \
        return sum;                                        \
    }                                                      \
                                                           \
    [[nodiscard]] auto mag() const noexcept -> double {    \
        return std::sqrt(this->mag_squared());             \
    }

#define NORM(T, Rows, Cols)                                                                       \
    auto norm() const noexcept -> std::expected<Mat<Rows, Cols, T>, Error> {                      \
        double mag = this->mag();                                                                 \
        if (mag == 0) {                                                                           \
            return std::unexpected(Error::ZeroDivision);                                          \
        }                                                                                         \
        auto normalized = this->data;                                                             \
        std::transform(this->data.begin(), this->data.end(), normalized.begin(),                  \
                       [mag](const T element) { return element / mag; });                         \
                                                                                                  \
        return Mat<Cols, Rows>(normalized);                                                       \
    }                                                                                             \
                                                                                                  \
    auto direction(const Mat &other) const noexcept -> std::expected<Mat<Rows, Cols, T>, Error> { \
        return other.sub(*this).norm();                                                           \
    }
}  // namespace vector

#define MATRIX(T, Rows, Cols)                 \
    ARITHMETIC(T, Rows, Cols, +, add, +)      \
    ARITHMETIC(T, Rows, Cols, -, sub, -)      \
    ARITHMETIC(T, Rows, Cols, *, hadamard, &) \
    MUL(T, Rows, Cols)                        \
    MUL_CONSTANT(T, Rows, Cols)               \
    TRANSPOSE(T, Rows, Cols)                  \
    CUT(T, Rows, Cols)                        \
    PAD(T, Rows, Cols)                        \
    ROW_COL(T, Rows, Cols)

#define VECTOR(T, Rows, Cols) \
    MATRIX(T, Rows, Cols)     \
    DOT(T, Rows, Cols)        \
    DOT(T, Cols, Rows)        \
    MAG(T, Rows, Cols)        \
    NORM(T, Rows, Cols)

#define THROW_OUT_OF_RANGE(i)                                                     \
    std::stringstream error_message;                                              \
    error_message << "array::at: __n (which is " << i << ") >= _Nm (which is 1)"; \
    throw std::out_of_range(error_message.str());

template <uint Rows, uint Cols, typename T = int>
using table = std::array<std::array<T, Cols>, Rows>;

template <uint Rows, uint Cols, typename T = int>
auto tablify(std::span<T> s) noexcept -> table<Rows, Cols, T> {
    table<Rows, Cols, T> data{};
    for (uint i = 0; auto &row : data) {
        auto sub = s.subspan(i, Cols);
        std::copy(sub.begin(), sub.end(), row.begin());
        i += Cols;
    }
    return data;
}

template <uint Rows, uint Cols, typename T = int>
auto filled(T fill) -> std::array<T, Rows * Cols> {
    std::array<T, Rows * Cols> repeat{};
    repeat.fill(fill);
    return std::move(repeat);
}
}  // namespace detail

enum class Error {
    OutOfRange,
    ZeroDivision,
};

template <uint Rows, uint Cols, uint R, uint C>
concept Extendable = ((R >= Rows and C >= Cols) and not(R == Rows and C == Cols));

template <uint Rows, uint Cols, uint R, uint C>
concept Shrinkable = ((R <= Rows and C <= Cols) and not(R == Rows and C == Cols));

template <typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, bool> = true>
concept TrivialNumber = std::convertible_to<T, double>;

/**
 * Common interface shared by all Mat classes.
 * @tparam Rows
 * @tparam Cols
 * @tparam T
 */
template <uint Rows, uint Cols, typename T = int>
class Interface {
    //    template <uint, uint, typename>
    //    friend class Mat;

   public:
    virtual auto at(uint, uint) -> T & = 0;
    virtual auto get(uint, uint) const noexcept -> std::expected<T, Error> = 0;

    [[nodiscard]] consteval auto size() const noexcept -> uint {
        return Cols * Rows;
    }

    /**
     * Getter/Setter for a specified element in the matrix.
     */
    auto operator[](uint i, uint j) -> T & {
        return this->at(i, j);
    };

    /**
     * Getter/Setter for an nth element in the matrix, disregarding dimensions.
     */
    auto operator[](uint i) -> T & {
        return this->at(i / Cols, i % Cols);
    }

    auto minmax() const noexcept -> std::pair<T, T> {
        T max{}, min{};
        for (uint i = 0; i < Rows; ++i) {
            for (uint j = 0; j < Cols; ++j) {
                if (T value = this->get(i, j).value(); value > max) {
                    max = value;
                } else if (value < min) {
                    min = value;
                }
            }
        }
        return std::make_pair(min, max);
    }

    friend auto operator<<(std::ostream &os, const Interface &mat) -> std::ostream & {
        auto [min, max] = mat.minmax();
        auto width = std::to_string(std::abs(min) > max ? min : max).size();

        os << "[";
        if (Rows != 1) {
            os << "[";
        }

        for (uint i = 0; i != Rows; i++) {
            for (uint j = 0; j != Cols; j++) {
                os << std::left << std::setw(width) << mat.get(i, j).value();
                if (j + 1 < Cols) {
                    os << " ";
                }
            }
            os << "]";
            if (i + 1 < Rows) {
                os << "\n [";
            }
        }

        if (Rows != 1) {
            os << "]";
        }
        return os;
    }

    /**
     * The == operator is implemented on all matrix specialization and matrices of different sizes,
     * it checks if the size if the same and then compares every element to each other.
     */
    template <uint R, uint C>
    auto operator==(const Interface<R, C, T> &mat) const noexcept -> bool {
        if (Rows * Cols != R * C) {
            return false;
        }

        for (uint i = 0; i < Rows; ++i) {
            for (uint j = 0; j < Rows; ++j) {
                if (this->get(i, j).value() != mat.get(i, j).value()) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * != is the negation of ==, refer to == for implementation details.
     */
    template <uint R, uint C>
    auto operator!=(const Interface<R, C, T> &mat) const noexcept -> bool {
        return not(*this == mat);
    }
};

//====================================================================================================
/**
 * The primary Mat class, includes all matrices NxM, where N>=1, and N!=M.
 * @tparam Rows
 * @tparam Cols
 * @tparam T
 *
 * The Mat class is a static sized matrix-like object with compile-time checks on parameters and numerous
 * operations. It features specializations for square matrices and vectors (1xN matrices) with their own
 * unique functions. Every single operation is on copy basis, which means that calling a function will almost
 * always copy the necessary contents for the operation from the callee to the caller. This is because the
 * underlying type for Mat is trivially-copyable and held on stack, so moving wouldn't provide any benefits
 * (according to the internet and Clang-Tidy).
 */
template <uint Rows, uint Cols, TrivialNumber T = int>
    requires(Rows >= 1 and Cols >= 1)
class Mat : public Interface<Rows, Cols, T> {
   protected:
    detail::table<Rows, Cols, T> data;

   public:
    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::array<T, Cols * Rows> array) : data(detail::tablify<Rows, Cols, T>(array)) {}
    explicit Mat(std::span<T> span) : data(detail::tablify<Rows, Cols, T>(span)) {}

    auto get(uint i, uint j) const noexcept -> std::expected<T, Error> override {
        if (i >= Rows or j >= Cols) {
            return std::unexpected(Error::OutOfRange);
        }
        return this->data.at(i).at(j);
    }

    auto at(uint i, uint j) -> T & override {
        return this->data.at(i).at(j);
    }

    /**
     * Getter for row amount.
     */
    [[nodiscard]] consteval auto rows() noexcept -> uint {
        return Rows;
    }

    /**
     * Getter for column amount.
     */
    [[nodiscard]] consteval auto cols() noexcept -> uint {
        return Cols;
    }

    MATRIX(T, Rows, Cols)
};

//====================================================================================================
/**
 * Convenience function for creating a square matrix.
 */
template <uint N, typename T = int, typename... Ts>
auto square(Ts... values) -> Mat<N, N, T> {
    return Mat<N, N, T>({values...});
}

/**
 * Convenience function for constructing a square matrix filled with copies of a value.
 */
template <uint N, typename T = int>
auto square(T fill) -> Mat<N, N, T> {
    return Mat<N, N, T>(detail::filled<N, N, T>(fill));
}

/**
 * Convenience function for creating a square identity matrix.
 */
template <uint N, typename T = int>
auto identity() -> Mat<N, N, T> {
    Mat<N, N, T> mat{};
    for (uint i = 0; i < N; ++i) {
        mat[i, i] = 1;
    }
    return mat;
}

/**
 * Square matrix specialization.
 */
template <uint N, typename T>
class Mat<N, N, T> : public Interface<N, N, T> {
   protected:
    detail::table<N, N, T> data;

   public:
    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::array<T, N * N> array) : data(detail::tablify<N, N, T>(array)) {}
    explicit Mat(std::span<T> span) : data(detail::tablify<N, N, T>(span)) {}

    auto get(uint i, uint j) const noexcept -> std::expected<T, Error> override {
        if (i >= N or j >= N) {
            return std::unexpected(Error::OutOfRange);
        }
        return this->data.at(i).at(j);
    }

    auto at(uint i, uint j) -> T & override {
        return this->data.at(i).at(j);
    }

    /**
     * Getter for Cols/Rows size, better known as dimension of the matrix.
     */
    [[nodiscard]] consteval auto dim() noexcept -> uint {
        return N;
    }

    [[nodiscard]] auto det() noexcept -> uint {
        // TODO finish determinant algorithm
    }

    MATRIX(T, N, N)
};

//====================================================================================================
/**
 * Convenience function for constructing a Mat\<1, N>.
 */
template <uint N, typename T = int, typename... Ts>
auto vec(Ts... values) -> Mat<1, N, T> {
    return Mat<1, N, T>({values...});
}

/**
 * Convenience function for constructing a Mat\<1, N> filled with copies of a value.
 */
template <uint N, typename T = int>
auto vec(T fill) -> Mat<1, N, T> {
    return Mat<1, N, T>(detail::filled<1, N, T>(fill));
}

/**
 * Convenience function for constructing a Mat\<1, 2>.
 */
template <typename T = int>
auto vec2(T x, T y) -> Mat<1, 2, T> {
    return Mat<1, 2, T>(x, y);
}

/**
 * Convenience function for constructing a Mat\<1, 3>.
 */
template <typename T = int>
auto vec3(T x, T y, T z) -> Mat<1, 3, T> {
    return Mat<1, 3, T>(x, y, z);
}

/**
 * General vector specialization, Nx1 matrices are not considered vectors.
 * @tparam N - capacity of the vector
 * @tparam T
 */
template <uint N, typename T>
class Mat<1, N, T> : public Interface<1, N, T> {
   private:
    std::array<T, N> data;

   public:
    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::array<T, N> array) : data(array) {}
    explicit Mat(std::span<T> span) {
        std::copy(span.begin(), span.end(), data.begin());
    }

    auto get(uint i, uint j) const noexcept -> std::expected<T, Error> override {
        if (i != 0 or j >= N) {
            return std::unexpected(Error::OutOfRange);
        }
        return this->data.at(j);
    }

    auto at(uint i, uint j) -> T & override {
        if (i != 0 or j >= N) {
            THROW_OUT_OF_RANGE(i)
        }
        return this->data.at(j);
    }

    VECTOR(T, 1, N)
};

//====================================================================================================
/**
 * Vec2 specialization.
 */
template <typename T>
class Mat<1, 2, T> : public Interface<1, 2, T> {
   public:
    T x, y;

    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::span<T, 2> array) : x(array[0]), y(array[1]) {}
    Mat(T x, T y) : x(x), y(y) {}

    auto at(uint i, uint j) -> T & override {
        if (i != 0 or j > 1) {
            THROW_OUT_OF_RANGE(std::min(i, j))
        } else if (j == 0) {
            return x;
        } else {
            return y;
        }
    }

    auto get(uint i, uint j) const noexcept -> std::expected<T, Error> override {
        if (i != 0 or j > 1) {
            return std::unexpected(Error::OutOfRange);
        } else if (j == 0) {
            return x;
        } else {
            return y;
        }
    }

    VECTOR(T, 1, 2)
};

//====================================================================================================
/**
 * Vec3 specialization.
 */
template <typename T>
class Mat<1, 3, T> : public Interface<1, 3, T> {
   public:
    T x, y, z;

    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::span<T, 3> array) : x(array[0]), y(array[1]), z(array[2]) {}
    Mat(T x, T y, T z) : x(x), y(y), z(z) {}
    
    auto get(uint i, uint j) const noexcept -> std::expected<T, Error> override {
        if (i != 0 or j > 2) {
            return std::unexpected(Error::OutOfRange);
        } else if (j == 0) {
            return x;
        } else if (j == 1) {
            return y;
        } else {
            return z;
        }
    }

    auto at(uint i, uint j) -> T & override {
        if (i != 0 or j > 2) {
            THROW_OUT_OF_RANGE(std::min(i, j))
        } else if (j == 0) {
            return x;
        } else if (j == 1) {
            return y;
        } else {
            return z;
        }
    }

    VECTOR(T, 1, 3)
};
}  // namespace mat

#undef ARITHMETIC
#undef MUL
#undef MUL_CONSTANT
#undef TRANSPOSE
#undef CUT
#undef DOT
#undef MAG
#undef NORM
#undef MATRIX
#undef VECTOR
#undef THROW_OUT_OF_RANGE
