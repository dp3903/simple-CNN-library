#include <vector>
#include "Value.hpp"

class Layer{
    public:
        string label;

        Layer(){}
        Layer(string name){
            this->label = name;
        }

        virtual vector<double> forward(vector<double> input){
            throw runtime_error("Invalid input.");
        }

        virtual vector<vector<double>> forward(vector<vector<double>> input){
            throw runtime_error("Invalid input.");
        }
        
        virtual vector<double> back_prop(vector<double> grads) = 0;
        virtual void update_weights(double learning_rate = 0.1) = 0;
};

class Dense : public Layer{
    public:
        vector<vector<Value>> weights;
        vector<double> input;

        Dense(){}

        Dense(string label, int input_size, int output_size) : Layer(label){
            this->label = (label.size()!=0 ? label : "temp");
            this->weights = vector<vector<Value>>(output_size, vector<Value>(input_size));
            this->input = vector<double>(input_size);
            for(int i=0 ; i < weights.size() ; i++){
                for(int j=0 ; j < weights[i].size() ; j++){
                    weights[i][j] = Value("Dense: "+this->label+" param("+to_string(i)+','+to_string(j)+").", double(rand()) / RAND_MAX);
                }
            }
        }

        vector<double> forward(vector<double> input){
            if(!weights.size() || !input.size() || !(this->input.size())){
                throw runtime_error("Cannot forword on empty layer.");
            }
            if(this->input.size() != input.size())
                throw runtime_error("Input size does not match expected input size.");

            this->input = input;
            vector<double> op = vector<double>(weights.size(), 0);
            try{
                for(int i=0 ; i < weights.size() ; i++){
                    for(int j=0 ; j < weights[i].size() ; j++){
                        op[i] += (input[j] * weights[i][j].val);
                    }
                }
                return op;
            }
            catch(string s){
                throw runtime_error("Error in forwarding from dense layer: '"+this->label+"'.");
            }
        }

        vector<double> back_prop(vector<double> grads){
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads.");
            if(grads.size() != weights.size())
                throw runtime_error("Input grads size does not match no. of neurons.");

            vector<double> op = vector<double>(weights[0].size(), 0);
            for(int i=0 ; i<weights.size() ; i++){
                for(int j=0 ; j<weights[i].size() ; j++){
                    weights[i][j].grad += input[j] * grads[i];
                    op[j] += weights[i][j].val * grads[i];
                }
            }
            return op;
        }

        void update_weights(double learning_rate = 0.1){
            for(int i=0 ; i<weights.size() ; i++){
                for(int j=0 ; j<weights[i].size() ; j++){
                    weights[i][j].val -= learning_rate * weights[i][j].grad;
                    weights[i][j].grad = 0;
                }
            }
        }
};

class Softmax : public Layer{
    public: 
        vector<double> input;

        Softmax(){}

        Softmax(string label, int size){
            this->label = label;
            this->input = vector<double>(size);
        }

        vector<double> forward(vector<double> input){
            if(input.size() != this->input.size()){
                throw runtime_error("Invalid input size for softmax layer: "+this->label);
            }
            this->input = input;
            double sum = 0.0;
            vector<double> op = vector<double>(input.size());
            for(int i=0 ; i < input.size() ; i++){
                op[i] = exp(input[i]);
                sum += op[i];
            }
            for(int i=0 ; i < input.size() ; i++){
                op[i] = op[i] / sum;
            }
            return op;
        }

        vector<double> back_prop(vector<double> grads){
            return grads;
        }

        void update_weights(double lr = 0.1){}
};