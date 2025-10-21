#pragma once
#include <ostream>
#include <cmath>
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/polymorphic.hpp"

using namespace std;

class Value{
    public:
        string label;
        double val;
        double grad;

        Value(string label="", double val=0, double grad=0){
            this->val = val;
            this->grad = grad;
            this->label = label;
        }

        Value operator+(Value b){
            return Value("("+this->label+"+"+b.label+")", this->val + b.val, this->grad + b.grad);
        }
        Value operator-(Value b){
            return Value("("+this->label+"-"+b.label+")", this->val - b.val, this->grad + b.grad);
        }
        Value operator*(Value b){
            return Value("("+this->label+"*"+b.label+")", this->val * b.val, this->grad * b.grad);
        }
        Value operator/(Value b){
            return Value("("+this->label+"/"+b.label+")", this->val / b.val, this->grad * b.grad);
        }
        Value operator+(double b){
            return Value("("+this->label+"+"+to_string(b)+")", this->val + b, this->grad);
        }
        Value operator-(double b){
            return Value("("+this->label+"-"+to_string(b)+")", this->val - b, this->grad);
        }
        Value operator*(double b){
            return Value("("+this->label+"*"+to_string(b)+")", this->val * b, this->grad);
        }
        Value operator/(double b){
            return Value("("+this->label+"/"+to_string(b)+")", this->val / b, this->grad);
        }
        Value operator^(double p){
            return Value("("+this->label+'^'+to_string(p)+")", pow(this->val, p), 0);
        }

        void update(){
            val += grad;
        }

        string as_string() const{
            return ("Value(val: " + to_string(this->val) + ", grad: " + to_string(this->grad) + ")");
        }

        template <class Archive>
        void serialize(Archive& archive) {
            archive(
                CEREAL_NVP(label),      // Saves the label
                CEREAL_NVP(val)
            );
        }
};

// A simple struct to hold shape information
struct Shape {
    size_t channels = 0;
    size_t height = 0;
    size_t width = 0;

    size_t total_elements() const { return max(1,(int)channels) * max(1,(int)height) * width; }

    string as_string() const { return ("Shape(" + to_string(channels) + ", " + to_string(height) + ", " + to_string(width) + ")"); }

    template <class Archive>
    void serialize(Archive& archive) {
        archive(CEREAL_NVP(channels));
        archive(CEREAL_NVP(height));
        archive(CEREAL_NVP(width));
    }
};

ostream& operator<<(ostream& os, const Shape& s){
    return (os << s.as_string() ); 
}

ostream& operator<<(ostream& os, const Value& obj) {
    return (os << obj.as_string());
}

string operator*(string s, int x){
    if(x < 0)
        throw invalid_argument("string cannot be multiplied by negative integers.");
        
    string out = "";
    while(x--)
        out += s;
    return out;
}

string operator*(int x, string s){
    if(x < 0)
        throw invalid_argument("string cannot be multiplied by negative integers.");
        
    string out = "";
    while(x--)
        out += s;
    return out;
}
