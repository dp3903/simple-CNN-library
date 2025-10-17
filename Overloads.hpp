#include <vector>
#include <iostream>
#include "Value.hpp"
#include <variant>
#include <iomanip>
#include <type_traits>

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