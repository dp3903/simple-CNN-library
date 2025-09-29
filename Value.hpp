#include <ostream>
#include <cmath>
using namespace std;

class Value{
    public:
        string label;
        double val;
        double grad;

        Value(){
            this->val = 0;
            this->grad = 0;
        }

        Value(string label, double val=0, double grad=0){
            this->val = val;
            this->grad = grad;
            this->label = label;
        }

        Value operator+(Value b){
            return Value("("+this->label+"+"+b.label+")", this->val + b.val, 0);
        }
        Value operator-(Value b){
            return Value("("+this->label+"-"+b.label+")", this->val - b.val, 0);
        }
        Value operator*(Value b){
            return Value("("+this->label+"*"+b.label+")", this->val * b.val, 0);
        }
        Value operator/(Value b){
            return Value("("+this->label+"/"+b.label+")", this->val / b.val, 0);
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