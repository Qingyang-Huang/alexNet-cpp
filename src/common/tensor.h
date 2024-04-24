#ifndef TENOSR_H
#define TENOSR_H

#include <cstdlib>
#include <iostream>
#include <stdexcept>

template<typename T>
class Tensor {
private:
    T* data;
    size_t dim1, dim2, dim3;

public:
    // Constructor
    Tensor(size_t dim1, size_t dim2, size_t dim3) : dim1(dim1), dim2(dim2), dim3(dim3) {
        data = static_cast<T*>(malloc(dim1 * dim2 * dim3 * sizeof(T)));
        if (data == nullptr) {
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] = T();
        }
    }

    Tensor(size_t dim1, size_t dim2) : Tensor(dim1, dim2, 1) {}

    Tensor(size_t dim1) : Tensor(dim1, 1, 1) {}

    Tensor(size_t d1, size_t d2, size_t d3, std::initializer_list<T> list) : Tensor(d1, d2, d3) {
        std::copy(list.begin(), list.end(), data);
    }

    Tensor(size_t d1, size_t d2, std::initializer_list<T> list) : Tensor(d1, d2, 1) {
        std::copy(list.begin(), list.end(), data);
    }

    // Destructor
    ~Tensor() {
        free(data);
    }

    // Copy constructor
    Tensor(const Tensor& other) : dim1(other.dim1), dim2(other.dim2), dim3(other.dim3) {
        data = static_cast<T*>(malloc(dim1 * dim2 * dim3 * sizeof(T)));
        if (data == nullptr) {
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < dim1 * dim2; i++) {
            data[i] = other.data[i];
        }
    }

    // Move constructor
    Tensor(Tensor&& other) noexcept : data(other.data), dim1(other.dim1), dim2(other.dim2), dim3(other.dim3) {
        other.data = nullptr;
        other.dim1 = 0;
        other.dim2 = 0;
        other.dim3 = 0;
    }

    // Copy assignment operator
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            if (dim1 * dim2 * dim3 != other.dim1 * other.dim2 * other.dim3) {
                free(data);
                data = static_cast<T*>(malloc(other.dim1 * other.dim2 * other.dim3 * sizeof(T)));
                if (data == nullptr) {
                    throw std::bad_alloc();
                }
            }
            dim1 = other.dim1;
            dim2 = other.dim2;
            dim3 = other.dim3;
            for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            free(data);
            data = other.data;
            dim1 = other.dim1;
            dim2 = other.dim2;
            dim3 = other.dim3;
            other.data = nullptr;
            other.dim1 = 0;
            other.dim2 = 0;
            other.dim3 = 0;
        }
        return *this;
    }

    bool operator==(const Tensor& other) const {
        if (dim1 * dim2 * dim3 != other.dim1 * other.dim2 * other.dim3) {
            return false;
        }
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            if (data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const Tensor& other) const {
        return !(*this == other);
    }

    // Function to get shape
    std::array<size_t, 3> shape() const {
        return {dim1, dim2, dim3};
    }

    size_t size() const {
        return dim1 * dim2 * dim3;
    }

    void resize(size_t row, size_t col, size_t chan) const{
        if (row * col * chan != dim1 * dim2 * dim3) {
            throw std::invalid_argument("New shape must be compatible with the old shape");
        }
        dim1 = row;
        dim2 = col;
        dim3 = chan;
    }
    void resize(size_t row, size_t col){
        size_t size = dim1 * dim2 * dim3;
        dim1 = row;
        dim2 = col;
        dim3 = size / (row * col);
    }

    Tensor<T> reshape(size_t row, size_t col, size_t chan) const {
        if (row * col * chan != dim1 * dim2 * dim3) {
            throw std::invalid_argument("New dimensions must match the number of elements in the tensor.");
        }
        Tensor<T> result(row, col, chan);
        for (size_t idx = 0, i = 0; i < row; i++) {
            for (size_t j = 0; j < col; j++) {
                for (size_t k = 0; k < chan; k++) {
                    result(i, j, k) = (*this)(idx);
                    idx++;
                }
            }
        }
        return result;
    }

    // Function to access elements
    T& operator()(size_t row, size_t col, size_t chan) {
        if ((row * dim2 + col) * dim3 + chan >= dim1 * dim2 * dim3 || row < 0 || col < 0 || chan < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[(row * dim2 + col) * dim3 + chan];
    }

    const T& operator()(size_t row, size_t col, size_t chan) const {
        if ((row * dim2 + col) * dim3 + chan >= dim1 * dim2 * dim3 || row < 0 || col < 0 || chan < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[(row * dim2 + col) * dim3 + chan];
    }

    T& operator()(size_t row, size_t col) {
        if (row * dim2 + col >= dim1 * dim2 || row < 0 || col < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[row * dim2 + col];
    }

    const T& operator()(size_t row, size_t col) const {
        if (row * dim2 + col >= dim1 * dim2 || row < 0 || col < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[row * dim2 + col];
    }

    T& operator()(size_t idx) {
        if (idx >= dim1 * dim2 * dim3 || idx < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[idx];
    }

    const T& operator()(size_t idx) const {
        if (idx >= dim1 * dim2 * dim3 || idx < 0) {
            throw std::out_of_range("Index out of range");
        }
        return data[idx];
    }

    Tensor<T> row(size_t row) const {
        if (row >= dim1) {
            throw std::out_of_range("Index out of range");
        }
        Tensor<T> result(1, dim2, dim3);
        for (size_t i = 0; i < dim2; i++) {
            for (size_t j = 0; j < dim3; j++) {
                result(0, i, j) = (*this)(row, i, j);
            }
        }
        return result;
    }

    Tensor<T> col(size_t col) const {
        if (col >= dim2) {
            throw std::out_of_range("Index out of range");
        }
        Tensor<T> result(dim1, 1, dim3);
        for (size_t i = 0; i < dim1; i++) {
            for (size_t j = 0; j < dim3; j++) {
                result(i, 0, j) = (*this)(i, col, j);
            }
        }
        return result;
    }

    Tensor<T> chan(size_t chan) const {
        if (chan >= dim3) {
            throw std::out_of_range("Index out of range");
        }
        Tensor<T> result(dim1, dim2, 1);
        for (size_t i = 0; i < dim1; i++) {
            for (size_t j = 0; j < dim2; j++) {
                result(i, j, 0) = (*this)(i, j, chan);
            }
        }
        return result;
    }

    // Arithmetic operators
    Tensor<T> operator+(const Tensor& other) const {
        if (dim1 != other.dim1 || dim2 != other.dim2 || dim3 != other.dim3) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Tensor<T> operator-(const Tensor& other) const {
        if (dim1 != other.dim1 || dim2 != other.dim2 || dim3 != other.dim3) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    Tensor<T> operator*(const Tensor& other) const {
        if (dim1 != other.dim1 || dim2 != other.dim2 || dim3 != other.dim3) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    Tensor<T> operator*(const T scalar) const {
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    Tensor<T> operator/(const T scalar) const {
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = data[i] / scalar;
        }
        return result;
    }

    Tensor<T> operator-() const {
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = -data[i];
        }
        return result;
    }

    Tensor& operator+=(const Tensor& other) {
        if (dim1 != other.dim1 || dim2 != other.dim2 || dim3 != other.dim3) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    Tensor& operator+=(const T scalar) {
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] += scalar;
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        if (dim1 != other.dim1 || dim2 != other.dim2 || dim3 != other.dim3) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    Tensor& operator-=(const T scalar) {
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] -= scalar;
        }
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
        if (dim1 != other.dim1 || dim2 != other.dim2 || dim3 != other.dim3) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    Tensor& operator*=(const T scalar) {
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] *= scalar;
        }
        return *this;
    }

    Tensor& operator/=(const T scalar) {
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] /= scalar;
        }
        return *this;
    }

    

    T sum() const {
        T result = T();
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result += data[i];
        }
        return result;
    }

    T max() const {
        T result = data[0];
        for (size_t i = 1; i < dim1 * dim2 * dim3; i++) {
            if (data[i] > result) {
                result = data[i];
            }
        }
        return result;
    }


    Tensor<T> t() const {
        Tensor<T> result(dim2, dim1, dim3);
        for (size_t i = 0; i < dim1; i++) {
            for (size_t j = 0; j < dim2; j++) {
                for (size_t k = 0; k < dim3; k++) {
                    result(j, i, k) = (*this)(i, j, k);
                }
            }
        }
        return result;
    }

    Tensor<T> dot(const Tensor& other) {
        if (dim2 != other.dim1) {
            throw std::invalid_argument("Tensor dimensions do not match");
        }
        Tensor<T> result(dim1, other.dim2, dim3);
        for (size_t i = 0; i < dim1; i++) {
            for (size_t j = 0; j < other.dim2; j++) {
                for (size_t k = 0; k < dim3; k++) {
                    for (size_t l = 0; l < dim2; l++) {
                        result(i, j, k) += (*this)(i, l, k) * other(l, j, k);
                    }
                }
            }
        }
        return result;
    }

    Tensor<T> clone() const {
        Tensor<T> result(dim1, dim2, dim3);
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            result.data[i] = data[i];
        }
        return result;
    }

    void zeros() {
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] = T();
        }
    }

    void ones(){
        for (size_t i = 0; i < dim1 * dim2 * dim3; i++) {
            data[i] = 1;
        }
    }

    void print() const{
        for (size_t i = 0; i < dim1; i++) {
            for (size_t j = 0; j < dim2; j++) {
                for (size_t k = 0; k < dim3; k++) {
                    std::cout << (*this)(i, j, k) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

};

#endif // TENOSR_H