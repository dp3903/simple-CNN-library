#pragma once
#include <vector>
#include <iostream>
#include "Value.hpp"
#include <variant>
#include <iomanip>
#include <type_traits>

// Use type aliases for readability. Place them here for library-wide scope.
using Tensor1D = std::vector<double>;
using Tensor2D = std::vector<std::vector<double>>;
using Tensor3D = std::vector<std::vector<std::vector<double>>>;
using Tensor = variant<Tensor1D, Tensor2D, Tensor3D>;
using Data = pair<Tensor,Tensor>;
using Batch = vector<Data>;
double inf = std::numeric_limits<double>::infinity();
double negInf = -inf;

Tensor create_tensor(const Shape& shape, double initial_value = 0.0) {
    // Logic to decide which type of tensor to create based on the shape
    
    // Case 1: 1D Tensor (e.g., shape = {0, 0, 128})
    if (shape.width > 0 && shape.height == 0 && shape.channels == 0) {
        return Tensor1D(shape.width, initial_value);
    }
    // Case 2: 2D Tensor (e.g., shape = {0, 28, 28})
    else if (shape.width > 0 && shape.height > 0 && shape.channels == 0) {
        return Tensor2D(shape.height, Tensor1D(shape.width, initial_value));
    }
    // Case 3: 3D Tensor (e.g., shape = {3, 28, 28})
    else if (shape.width > 0 && shape.height > 0 && shape.channels > 0) {
        return Tensor3D(shape.channels, Tensor2D(shape.height, Tensor1D(shape.width, initial_value)));
    }
    // Handle invalid shape
    else {
        throw std::invalid_argument("Invalid shape provided to create_tensor.");
    }
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<T>
> 
operator+(vector<T> a, vector<T> b){
    if(a.size() != b.size())
        throw runtime_error("Incompatible sizes for addition of vectors: "+to_string(a.size())+"!="+to_string(b.size()));

    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] + b[i];
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<T>
> 
operator-(vector<T> a, vector<T> b){
    if(a.size() != b.size())
        throw runtime_error("Incompatible sizes for subtraction of vectors: "+to_string(a.size())+"!="+to_string(b.size()));

    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] - b[i];
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<T>
> 
operator+(vector<T> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] + b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<T>
> 
operator-(vector<T> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] - b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<T>
> 
operator*(vector<T> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] * b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<T>
> 
operator/(vector<T> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] / b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<T>>
> 
operator+(vector<vector<T>> a, vector<vector<T>> b){
    if(a.size() != b.size())
        throw runtime_error("Incompatible sizes for addition of matrices: "+to_string(a.size())+"!="+to_string(b.size()));

    for(int i=0 ; i<a.size() ; i++){
        a[i] = a[i] + b[i];
    }
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<T>>
> 
operator-(vector<vector<T>> a, vector<vector<T>> b){
    if(a.size() != b.size())
        throw runtime_error("Incompatible sizes for subtraction of matrices: "+to_string(a.size())+"!="+to_string(b.size()));

    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] - b[i];
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<T>>
> 
operator+(vector<vector<T>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] + b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<T>>
> 
operator-(vector<vector<T>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] - b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<T>>
> 
operator*(vector<vector<T>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] * b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<T>>
> 
operator/(vector<vector<T>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] / b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<vector<T>>>
> 
operator+(vector<vector<vector<T>>> a, vector<vector<vector<T>>> b){
    if(a.size() != b.size())
        throw runtime_error("Incompatible sizes for addition of Tensors: "+to_string(a.size())+"!="+to_string(b.size()));

    for(int i=0 ; i<a.size() ; i++){
        a[i] = a[i] + b[i];
    }
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<vector<T>>>
> 
operator-(vector<vector<vector<T>>> a, vector<vector<vector<T>>> b){
    if(a.size() != b.size())
        throw runtime_error("Incompatible sizes for subtraction of Tensors: "+to_string(a.size())+"!="+to_string(b.size()));

    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] - b[i];
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<vector<T>>>
> 
operator+(vector<vector<vector<T>>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] + b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<vector<T>>>
> 
operator-(vector<vector<vector<T>>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] - b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<vector<T>>>
> 
operator*(vector<vector<vector<T>>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] * b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<vector<T>>>
> 
operator/(vector<vector<vector<T>>> a, double b){
    for(int i=0 ; i<a.size() ; i++)
        a[i] = a[i] / b;
    
    return a;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    ostream&
>
operator<<(ostream& os, const vector<T>& obj) {
    os<<"vector: [  ";
    for(T x: obj)
        os<< std::scientific << std::setprecision(4)<<x<<", ";
    os<<"\b\b ]";
    return os;
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, int> || 
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    ostream&
>
operator<<(ostream& os, const vector<vector<T>>& obj) {
    os<<"matrix: [ \n";
    for(vector<T> x: obj)
        os<<' '<<x<<",\n";
    os<<"\b\b ]";
    return os;
}


// --- The operator+ overload ---
Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    // std::visit takes a visitor (our lambda) and one or more variants.
    // It calls the lambda with the contained values of the variants.
    return std::visit([&](const auto& a, const auto& b) -> Tensor {
        // Use a compile-time if to check if the two variants hold the same type.
        if constexpr (std::is_same_v<decltype(a), decltype(b)>) {
            return a+b;
        } else {
            // If the types are different (e.g., Tensor1D + Tensor2D),
            // this is a compile-time error, so we throw an exception.
            throw std::invalid_argument("Cannot add tensors of different dimensions.");
        }
    }, lhs, rhs);
}

// --- The operator+ overload ---
Tensor operator+(const Tensor& lhs, const double& rhs) {
    // std::visit takes a visitor (our lambda) and one or more variants.
    // It calls the lambda with the contained values of the variants.
    return std::visit([&](const auto& a) -> Tensor {
        return a+rhs;
    }, lhs);
}

// --- The operator- overload ---
Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    // std::visit takes a visitor (our lambda) and one or more variants.
    // It calls the lambda with the contained values of the variants.
    return std::visit([](const auto& a, const auto& b) -> Tensor {
        // Use a compile-time if to check if the two variants hold the same type.
        if constexpr (std::is_same_v<decltype(a), decltype(b)>) {
            return a-b;
        } else {
            // If the types are different (e.g., Tensor1D - Tensor2D),
            // this is a compile-time error, so we throw an exception.
            throw std::invalid_argument("Cannot subtract tensors of different dimensions.");
        }
    }, lhs, rhs);
}

// --- The operator- overload ---
Tensor operator-(const Tensor& lhs, const double& rhs) {
    // std::visit takes a visitor (our lambda) and one or more variants.
    // It calls the lambda with the contained values of the variants.
    return std::visit([&](const auto& a) -> Tensor {
        return a-rhs;
    }, lhs);
}

// --- The operator* overload ---
Tensor operator*(const Tensor& lhs, const double& rhs) {
    // std::visit takes a visitor (our lambda) and one or more variants.
    // It calls the lambda with the contained values of the variants.
    return std::visit([&](const auto& a) -> Tensor {
        return a*rhs;
    }, lhs);
}


// --- The operator/ overload ---
Tensor operator/(const Tensor& lhs, const double& rhs) {
    // std::visit takes a visitor (our lambda) and one or more variants.
    // It calls the lambda with the contained values of the variants.
    return std::visit([&](const auto& a) -> Tensor {
        return a/rhs;
    }, lhs);
}

// --- Function to get the total number of elements in a tensor ---
size_t size(const Tensor& t) {
    return std::visit([](const auto& tensor) -> size_t {
        // This recursive lambda is the visitor that will count the elements.
        auto op = [&](auto& self, const auto& v) -> size_t {
            // Base case: If the elements inside the container are doubles,
            // we've reached an innermost vector. Return its size.
            if constexpr (std::is_same_v<typename std::decay_t<decltype(v)>::value_type, double>) {
                return v.size();
            } else {
                // Recursive step: Sum the sizes of the inner elements.
                size_t total = 0;
                for (const auto& elem : v) {
                    total += self(self, elem);
                }
                return total;
            }
        };
        // Initial call to the recursive lambda with the concrete tensor type
        return op(op, tensor);
    }, t);
}

template <typename T>
std::enable_if_t<
    std::is_same_v<T, double> || 
    std::is_same_v<T, Value>,
    vector<vector<double>>
>
convolve(
    vector<vector<double>> mat, 
    vector<vector<T>> filter, 
    int padding = 0, 
    int stride = 1, 
    int dilation = 1
){
    // 1. Get dimensions
    int input_H = mat.size();
    int input_W = mat[0].size();
    int filter_H = filter.size();
    int filter_W = filter[0].size();

    // 2. Calculate output dimensions (standard formula)
    // H_out = floor((H_in - H_filter + 2*P) / S) + 1
    // Note: Dilation is already accounted for in the H_filter term (H_filter_effective = D * (H_filter - 1) + 1)
    // For non-unit dilation, the formula should use the effective filter size.
    int effective_filter_H = dilation * (filter_H - 1) + 1;
    int effective_filter_W = dilation * (filter_W - 1) + 1;

    int out_H = (input_H - effective_filter_H + 2 * padding) / stride + 1;
    int out_W = (input_W - effective_filter_W + 2 * padding) / stride + 1;

    // Check for invalid dimension, though integer division in C++ often handles this.
    if (out_H <= 0 || out_W <= 0) {
        return {}; // Return empty matrix if output dimensions are invalid
    }

    // 3. Initialize the output matrix (Height, then Width)
    vector<vector<double>> ans(out_H, vector<double>(out_W));

    // 4. Perform Convolution
    // out_y = Row (Height), out_x = Column (Width)
    for (int out_y = 0; out_y < out_H; out_y++) {
        for (int out_x = 0; out_x < out_W; out_x++) {
            double sum = 0;
            
            // Loop over filter rows (i) and columns (j)
            for (int i = 0; i < filter_H; i++) {
                for (int j = 0; j < filter_W; j++) {
                    
                    // Calculate corresponding input coordinates (y, x)
                    // in_coord = out_coord * Stride + Dilation * Filter_coord - Padding
                    int in_y = out_y * stride + dilation * i - padding;
                    int in_x = out_x * stride + dilation * j - padding;
                    
                    // Boundary Check
                    if (in_y >= 0 && in_y < input_H && 
                        in_x >= 0 && in_x < input_W) 
                    {
                        // Standard Convolution: Filter element * Input element
                        // Note the correct [row/y][column/x] indexing for mat
                        if constexpr(std::is_same_v<T, Value>)
                            sum += filter[i][j].val * mat[in_y][in_x];                                    
                        else
                            sum += filter[i][j] * mat[in_y][in_x];
                    }
                    // If out of bounds, padding assumes a zero, so we do nothing.
                }
            }
            ans[out_y][out_x] = sum;
        }
    }

    return ans;
}