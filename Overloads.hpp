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
    return std::visit([](const auto& a, const auto& b) -> Tensor {
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