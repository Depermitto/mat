#pragma once

#include <cmath>
#include <format>
#include <functional>
#include <iomanip>
#include <numeric>
#include <ranges>

namespace mat {

template <typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, bool> = true>
concept TriviallyNumeric = std::convertible_to<T, double>;

template <uint Rows, uint Cols>
concept Vector = (Rows == 1 or Cols == 1);

template <TriviallyNumeric T, uint Rows, uint Cols>
class Mamut {
    using array = T[Rows * Cols];
    using iterator = T*;
    using const_iterator = const T*;

   protected:
    array data{};

    auto arithmetic_operation(const Mamut& other, std::function<T(T, T)>&& f) -> Mamut& {
        for (auto [i, j] : std::views::zip(*this, other)) {
            i = f(i, j);
        }
        return *this;
    }

   public:
    //! Copy the elements from data to create a new Mamut matrix.
    //! \param data Variable [0, Rows * Cols] amount of TriviallyNumeric arguments.
    explicit Mamut(TriviallyNumeric auto... data) : data(data...) {}
    //! Copy the elements from the begin to end and create a new Mamut matrix out of them.
    //! \param begin
    //! \param end
    Mamut(std::forward_iterator auto begin, std::forward_iterator auto end) {
        std::copy(begin, end, this->begin());
    }
    //! Copy the first n elements from begin iterator and create a new Mamut matrix out of them.
    //! \param begin
    //! \param n Defaults to Rows * Cols.
    explicit Mamut(std::forward_iterator auto begin, size_t n = Rows * Cols) {
        std::copy_n(begin, n, this->begin());
    }

    //! After constructing the default Mamut, set diagonal elements to 1.
    //! @defgroup Square matrix
    //! \return Mamut matrix.
    static auto identity() -> Mamut
        requires(Rows == Cols)
    {
        auto mamut = Mamut<T, Rows, Cols>();
        for (uint i = 0; i < Rows; i++) {
            mamut[i, i] = 1;
        }
        return mamut;
    }

    auto begin() -> iterator {
        return std::begin(data);
    }
    auto begin() const -> const_iterator {
        return std::cbegin(data);
    }

    auto end() -> iterator {
        return std::end(data);
    }
    auto end() const -> const_iterator {
        return std::cend(data);
    }

    //! Extends of the matrix in a std::pair.
    //! \return {Rows, Cols}.
    constexpr auto extends() const -> std::pair<uint, uint> {
        return std::make_pair(Rows, Cols);
    }
    //! Total amount of elements in the matrix.
    //! \return Rows * Cols.
    constexpr auto size() const -> uint {
        return Rows * Cols;
    }

    //! Getter for the ith element in the matrix.
    //! \param i ith element to get.
    //! \return std::nullopt if not found.
    auto get(uint i) const -> std::optional<T> {
        if (i >= size()) {
            return std::nullopt;
        }
        return *(begin() + i);
    }
    //! Getter for the [i, j] element in the matrix.
    //! \param i Row index.
    //! \param j Column index.
    //! \return std::nullopt if not found.
    auto get(uint i, uint j) const -> std::optional<T> {
        auto const [_, c] = extends();
        return get(i * c + j);
    }

    //! Getter for the reference of the ith element in the matrix.
    //! \param i ith element to get.
    //! \return Throws if out of range.
    auto at(uint i) -> T& {
        if (i >= size()) {
            throw std::out_of_range(std::format("array index [{}] out of index bound [{}]", i, size()));
        }
        return *(begin() + i);
    }
    //! Getter for the reference of the [i, j] element in the matrix.
    //! \param i Row index.
    //! \param j Column index.
    //! \return Throws if out of range.
    auto at(uint i, uint j) -> T& {
        auto const [r, c] = extends();
        if (i >= r or j >= c) {
            throw std::out_of_range(
                std::format("array index [{}, {}] out of index bounds [{}, {}]", i, j, r - 1, c - 1));
        }
        return *(begin() + i * c + j);
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

    //! Clones the contents of the matrix. Use this function if you wish for mutating
    //! methods to not mutate the caller.
    //! \return Clone of the Mamut.
    auto clone() const -> Mamut {
        return *this;
    }

    //! Performs standard matrix multiplication. Only accepts matrices that have amount of columns
    //! matching current Mamut's amount of rows.
    //! @example Mamut[2, 7] * Mamut[7, 2]
    //! @example Mamut[6, 8] * Mamut[8, 1]
    //! \tparam R Other Mamut's amount of rows.
    //! \param other Other Mamut.
    //! \return Mamut[current Mamut amount of rows, other Mamut amount of cols]
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
    //! Performs multiplication of all Mamut elements by a constant. Mutates the caller.
    //! \param constant Must be TriviallyNumeric.
    //! \return Reference of mutated caller.
    auto mul(TriviallyNumeric auto constant) -> Mamut& {
        std::for_each(begin(), end(), [constant](T& i) { i *= constant; });
        return *this;
    }

    //! Performs addition of other elements to caller elements. Mutates the caller.
    //! \operators + and +=
    //! @note Operators may act a little bit differently that its origin method. Please read up on them!
    //! \param other Must have same extends.
    //! \return Reference of mutated caller.
    auto add(const Mamut& other) -> Mamut& {
        return arithmetic_operation(other, std::plus<T>());
    }
    //! Performs addition of other elements to caller elements. Copies the caller.
    //! \param other Must have same extends.
    //! \return Result of addition.
    auto operator+(const Mamut& other) const -> Mamut {
        return this->clone().add(other);
    }
    //! Performs addition of other elements to caller elements. Mutates the caller.
    //! \param other Must have same extends.
    auto operator+=(const Mamut& other) -> void {
        this->add(other);
    }

    //! Performs subtraction of other elements from caller elements. Mutates the caller.
    //! \operators - and -=
    //! @note Operators may act a little bit differently that its origin method. Please read up on them!
    //! \param other Must have same extends.
    //! \return Reference of mutated caller.
    auto sub(const Mamut& other) -> Mamut& {
        return arithmetic_operation(other, std::minus<T>());
    }
    //! Performs subtraction of other elements from caller elements. Copies the caller.
    //! \param other Must have same extends.
    //! \return Result of subtraction.
    auto operator-(const Mamut& other) const -> Mamut {
        return this->clone().sub(other);
    }
    //! Performs subtraction of other elements from caller elements. Mutates the caller.
    //! \param other Must have same extends.
    auto operator-=(const Mamut& other) -> void {
        this->sub(other);
    }

    //! Performs hadamard multiplication of other elements and caller elements. Mutates the caller.
    //! \operators & and &=
    //! @note Operators may act a little bit differently that its origin method. Please read up on them!
    //! \param other Must have same extends.
    //! \return Reference of mutated caller.
    auto hadamard(const Mamut& other) -> Mamut& {
        return arithmetic_operation(other, std::multiplies<T>());
    }
    //! Performs hadamard multiplication of other elements and caller elements. Copies the caller.
    //! \param other Must have same extends.
    //! \return Result of hadamard.
    auto operator&(const Mamut& other) const -> Mamut {
        return this->clone().hadamard(other);
    }
    //! Performs hadamard multiplication of other elements and caller elements. Mutates the caller.
    //! \param other Must have same extends.
    auto operator&=(const Mamut& other) -> void {
        this->hadamard(other);
    }

    //! Transposes the copy of the caller. Elements stay the same.
    //! \operators ~
    //! \return Mamut[Rows, Cols] -> Mamut[Cols, Rows]
    auto transpose() const -> Mamut<T, Cols, Rows> {
        auto transposed = Mamut<T, Cols, Rows>();
        for (uint i = 0; i < Rows; ++i) {
            for (uint j = 0; j < Cols; ++j) {
                transposed[j, i] = this->get(i, j).value();
            }
        }
        return transposed;
    }
    auto operator~() const -> Mamut<T, Cols, Rows> {
        return this->transpose();
    }

    //! Trims the matrix into target size. Any excess elements that don't find into
    //! the new matrix will be discarded. Cannot trim into the same size as caller.
    //! \tparam R Target Rows. Must be smaller or equal than current Mamut amount of rows.
    //! \tparam C Target Cols. Must be smaller or equal than current Mamut amount of columns.
    //! \return Trimmed copy of the caller.
    template <uint R, uint C>
        requires((R <= Rows and C <= Cols) and not(R == Rows and C == Cols))
    auto cut() const -> Mamut<T, R, C> {
        return Mamut<T, R, C>(begin(), begin() + R * C);
    }

    //! Expands the matrix into target size. Any elements will be copied over. Cannot
    //! expand into the same size as caller.
    //! \tparam R Target Rows. Must be greater or equal than current Mamut amount of rows.
    //! \tparam C Target Cols. Must be greater or equal than current Mamut amount of columns.
    //! \return Expanded copy of the caller.
    template <uint R, uint C>
        requires((R >= Rows and C >= Cols) and not(R == Rows and C == Cols))
    auto pad() const -> Mamut<T, R, C> {
        return Mamut<T, R, C>(begin(), end());
    }

    //! Resizes the matrix into target size. Any excess elements will be discarded.
    //! \tparam R Target Rows.
    //! \tparam C Target Cols.
    //! \return Resized copy of the caller, the return value is similar to calling .pad().cut(),
    //! albeit more efficient.
    template <uint R, uint C>
    auto resize_to() const -> Mamut<T, R, C> {
        if constexpr (R * C == Rows * Cols) {
            return *this;
        }
        return Mamut<T, R, C>(begin(), begin() + std::min(R * C, Rows * Cols));
    }

    //! Get copy of specified row.
    //! \tparam R Index of row to get.
    //! \return Constructed vector-like Mamut.
    template <uint R>
        requires(R < Rows)
    auto row() const -> Mamut<T, 1, Cols> {
        return Mamut<T, 1, Cols>(begin() + R * Cols, Cols);
    }

    //! Construct a new row (as-if constructing a new vector-like Mamut by using a valid constructor)
    //! and replace a specified row with it.
    //! \tparam R Index of the row to mutate.
    //! \param constructor_args Arguments to be provided for the constructor of the new row.
    //! \return Mutated caller. This is to allow chaining methods.
    template <uint R>
        requires(R < Rows)
    auto mod_row(auto&&... constructor_args) -> Mamut& {
        auto row = Mamut<T, 1, Cols>(constructor_args...);
        for (uint i = 0; i < Cols; ++i) {
            this->at(R, i) = row[0, i];
        }
        return *this;
    }

    //! Get copy of specified column.
    //! \tparam C Index of column to get.
    //! \return Constructed vector-like Mamut.
    template <uint C>
        requires(C < Cols)
    auto col() const -> Mamut<T, Rows, 1> {
        Mamut<T, Rows, 1> col{};
        for (uint j = 0; j < Rows; ++j) {
            col[j, 0] = this->get(j, C).value();
        }
        return col;
    }

    //! Construct a new column (as-if constructing a new vector-like Mamut by using a valid constructor)
    //! and replace a specified column with it.
    //! \tparam C Index of the column to mutate.
    //! \param constructor_args Arguments to be provided for the constructor of the new column.
    //! \return Mutated caller. This is to allow chaining methods.
    template <uint C>
        requires(C < Cols)
    auto mod_col(auto&&... constructor_args) -> Mamut& {
        auto col = Mamut<T, Rows, 1>(constructor_args...);
        for (uint j = 0; j < Rows; ++j) {
            this->at(j, C) = col[j, 0];
        }
        return *this;
    }

    // Vectors

    //! @defgroup Vectors
    //! \return Squared magnitude of the vector.
    auto mag_squared() const -> T
        requires Vector<Rows, Cols>
    {
        return std::accumulate(begin(), end(), T(), [](auto sum, auto i) { return sum + std::pow(i, 2); });
    }

    //! @defgroup Vectors
    //! \return Magnitude of the vector.
    auto mag() const -> double
        requires Vector<Rows, Cols>
    {
        return std::sqrt(mag_squared());
    }

    //! Normalize the copy of caller and return it. This methods calls mag().
    //! @defgroup Vectors
    //! \return std::nullopt if mag() is 0.
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

    //! Calculate the direction from caller to other.
    //! @defgroup Vectors
    //! \param other Vector-like Mamut. Accepts greater dimensions than caller.
    //! \return std::nullopt if mag() is 0.
    template <uint R, uint C>
    auto direction(const Mamut<T, R, C>& other) const -> std::optional<Mamut<double, R, C>>
        requires Vector<Rows, Cols> and Vector<R, C> and (R * C >= Rows * Cols)
    {
        return (other - this->resize_to<R, C>()).norm();
    }

    //! Performs standard dot operation between two vector-like Mamuts. Accepts same sized
    //! vectors, as well as transposed vectors.
    //! @defgroup Vectors
    //! @example Mamut[1, 4] ^ Mamut[1, 4]
    //! @example Mamut[1, 4] ^ Mamut[4, 1]
    //! \operators ^
    //! \tparam R Amount of rows of other.
    //! \tparam C Amount of columns of other.
    //! \return Result of dot multiplication.
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
    //! Performs standard dot operation between two vector-like Mamuts. Accepts same sized
    //! vectors, as well as transposed vectors.
    //! @defgroup Vectors
    //! @example Mamut[1, 4] ^ Mamut[1, 4]
    //! @example Mamut[1, 4] ^ Mamut[4, 1]
    //! \tparam R Amount of rows of other.
    //! \tparam C Amount of columns of other.
    //! \return Result of dot multiplication.
    template <uint R, uint C>
    auto operator^(const Mamut<T, R, C>& other) const -> T
        requires Vector<Rows, Cols> and Vector<R, C> and (R * C == Rows * Cols)
    {
        return this->dot(other);
    }
};
}  // namespace mat
