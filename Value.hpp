#include <ostream>
#include <cmath>
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
};

ostream& operator<<(ostream& os, const Value& obj) {
    return (os << "Value(val: " << obj.val << ", grad: " << obj.grad << ")");
}