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

        string as_string() const{
            return ("Value(val: " + to_string(this->val) + ", grad: " + to_string(this->grad) + ")");
        }
};

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
