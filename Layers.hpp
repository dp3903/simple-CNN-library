#include <vector>
#include "Value.hpp"

class Layer{
    public:
        string label;

        Layer(){}
        Layer(string name){
            this->label = name;
        }

        virtual vector<Value> forward(vector<Value> input){
            throw runtime_error("Invalid input.");
        }

        virtual vector<vector<Value>> forward(vector<vector<Value>> input){
            throw runtime_error("Invalid input.");
        }
};

class Dense : public Layer{
    public:
        vector<vector<double>> weights;
        vector<Value> values;

        Dense(){}

        Dense(string label, int input_size, int output_size) : Layer(label){
            if(weights.size() != values.size()){
                throw runtime_error("Size of weights vector must match no. of values.");
            }
            this->label = (label.size()!=0 ? label : "temp");
            this->weights = vector<vector<double>>(output_size, vector<double>(input_size));
            this->values = vector<Value>(output_size);
            for(int i=0 ; i < weights.size() ; i++){
                this->values[i] = Value(label+" param:"+to_string(i));
                for(int j=0 ; j < weights[i].size() ; j++){
                    weights[i][j] = double(rand()) / RAND_MAX;
                }
            }
        }

        vector<Value> forward(vector<Value> input){
            if(!weights.size() || !values.size()){
                throw runtime_error("Cannot forword on empty layer.");
            }

            try{
                for(int i=0 ; i < weights.size() ; i++){
                    values[i].val = 0.0;
                    for(int j=0 ; j < weights[i].size() ; j++){
                        values[i].val += (input[j].val * weights[i][j]);
                    }
                }
                return values;
            }
            catch(string s){
                throw runtime_error("Error in forwarding from dense layer: '"+this->label+"'.");
            }
        }
};

class Softmax : public Layer{
    public: 
        vector<Value> values;

        Softmax(){}

        Softmax(string label, int size){
            this->label = label;
            this->values = vector<Value>(size);
        }

        vector<Value> forward(vector<Value> input){
            if(input.size() != values.size()){
                throw runtime_error("Invalid input size for softmax layer: "+this->label);
            }
            double sum = 0.0;
            for(int i=0 ; i < input.size() ; i++){
                input[i].val = exp(input[i].val);
                sum += input[i].val;
            }
            for(int i=0 ; i < input.size() ; i++){
                values[i].val = input[i].val / sum;
            }
            return values;
        }
};