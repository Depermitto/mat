#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
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
    template <typename Number, std::enable_if_t<std::is_arithmetic_v<Number>, bool> = true>     \
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
    template <typename Number, std::enable_if_t<std::is_arithmetic_v<Number>, bool> = true>     \
    friend auto operator*(const Mat<Rows, Cols, T> &mat, Number constant)->Mat<Rows, Cols, T> { \
        return mat.mul(constant);                                                               \
    }                                                                                           \
                                                                                                \
    template <typename Number, std::enable_if_t<std::is_arithmetic_v<Number>, bool> = true>     \
    friend auto operator*(Number constant, const Mat<Rows, Cols, T> &mat)->Mat<Rows, Cols, T> { \
        return mat.mul(constant);                                                               \
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
    }

#define CUT(T, Rows, Cols)                                                                   \
    /**                                                                                      \
     * Cut a submatrix from one of the corners.                                              \
     */                                                                                      \
    template <uint R, uint C>                                                                \
    auto cut(bool take_right = false, bool take_down = false) const noexcept -> Mat<R, C, T> \
        requires Shrinkable<Rows, Cols, R, C>                                                \
    {                                                                                        \
        Mat<R, C, T> shrunk{};                                                               \
        for (uint i = 0; i < R; ++i) {                                                       \
            for (uint j = 0; j < C; ++j) {                                                   \
                uint take_i = take_down ? (Rows - R) + i : i;                                \
                uint take_j = take_right ? (Cols - C) + j : j;                               \
                shrunk[i, j] = this->get(take_i, take_j).value();                            \
            }                                                                                \
        }                                                                                    \
        return shrunk;                                                                       \
    }                                                                                        \
                                                                                             \
    /**                                                                                      \
     * Cut submatrix from a specified location within the original matrix,                   \
     * starting point denoted by (i, j).                                                     \
     */                                                                                      \
    template <uint R, uint C>                                                                \
    auto submatrix(uint row, uint col) const noexcept -> std::expected<Mat<R, C, T>, Error>  \
        requires Shrinkable<Rows, Cols, R, C>                                                \
    {                                                                                        \
        Mat<R, C, T> shrunk{};                                                               \
        for (uint i = 0; i < R; ++i) {                                                       \
            for (uint j = 0; j < C; ++j) {                                                   \
                if (not this->get(i + row, j + col).has_value()) {                           \
                    return std::unexpected(Error::OutOfRange);                               \
                }                                                                            \
                shrunk[i, j] = this->get(i + row, j + col).value();                          \
            }                                                                                \
        }                                                                                    \
        return shrunk;                                                                       \
    }                                                                                        \
    /**                                                                                      \
     * Extend the matrix by padding new elements with the default value.                     \
     */                                                                                      \
    template <uint R, uint C>                                                                \
    auto extend() const noexcept -> Mat<R, C, T>                                             \
        requires Extendable<Rows, Cols, R, C>                                                \
    {                                                                                        \
        Mat<R, C, T> mat{};                                                                  \
        for (uint i = 0; i < Rows; ++i) {                                                    \
            for (uint j = 0; j < Cols; ++j) {                                                \
                mat[i, j] = this->get(i, j).value();                                         \
            }                                                                                \
        }                                                                                    \
        return mat;                                                                          \
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
    CUT(T, Rows, Cols)

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

    friend auto operator<<(std::ostream &os, const Interface &mat) -> std::ostream & {
        os << "[";
        if (Rows != 1) {
            os << "[";
        }

        for (uint i = 0; i != Rows; i++) {
            for (uint j = 0; j != Cols; j++) {
                os << mat.get(i, j).value();
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
 * Helper function for creating a square matrix.
 */
template <uint N, typename T = int, typename... U>
auto square(U... values) -> Mat<N, N, T> {
    return Mat<N, N, T>({values...});
}

/**
 * Helper function for creating a square identity matrix.
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
 * Helper function for constructing a Mat\<1, N>. Defaults to zero vector.
 */
template <uint N, typename T = int, typename... U>
auto vec(U... values) -> Mat<1, N, T> {
    return Mat<1, N, T>({values...});
}

/**
 * Helper function for constructing a Mat\<1, 2>.
 */
template <typename T = int>
auto vec2(T x, T y) -> Mat<1, 2, T> {
    return Mat<1, 2, T>(x, y);
}

/**
 * Helper function for constructing a Mat\<1, 3>.
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
   protected:
    auto get(uint i, uint j) const noexcept -> std::expected<T, Error> override {
        if (i != 0 or j > 1) {
            return std::unexpected(Error::OutOfRange);
        } else if (j == 0) {
            return x;
        } else {
            return y;
        }
    }

    auto at(uint i, uint j) -> T & override {
        if (i != 0 or j > 1) {
            THROW_OUT_OF_RANGE(std::min(i, j))
        } else if (j == 0) {
            return x;
        } else {
            return y;
        }
    }

   public:
    T x, y;

    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::array<T, 2> array) : x(array[0]), y(array[1]) {}
    Mat(T x, T y) : x(x), y(y) {}

    VECTOR(T, 1, 2)
};

//====================================================================================================
/**
 * Vec3 specialization.
 */
template <typename T>
class Mat<1, 3, T> : public Interface<1, 3, T> {
   protected:
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

   public:
    T x, y, z;

    Mat() = default;
    ~Mat() = default;
    explicit Mat(std::array<T, 3> array) : x(array[0]), y(array[1]), z(array[2]) {}
    Mat(T x, T y, T z) : x(x), y(y), z(z) {}

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
