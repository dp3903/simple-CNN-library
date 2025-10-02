#include <vector>
#include <iostream>
#include "Value.hpp"

double generate_random_in_range(double low, double high){
    if(high < low)
        swap(high,low);
    return (double(rand()) / RAND_MAX)*(high-low) + low;
}

class Layer{
    public:
        string label;

        Layer(){}
        Layer(string name){
            this->label = name;
        }

        virtual vector<double> forward(vector<double> input){
            throw runtime_error("Invalid input for layer: "+this->label+".");
        }

        virtual vector<vector<double>> forward(vector<vector<double>> input){
            throw runtime_error("Invalid input for layer: "+this->label+".");
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
                    weights[i][j] = Value("Dense: "+this->label+" param("+to_string(i)+','+to_string(j)+").", generate_random_in_range(-0.1, 0.1));
                }
            }
        }

        vector<double> forward(vector<double> input){
            if(!weights.size() || !input.size() || !(this->input.size())){
                throw runtime_error("Cannot forword on empty layer for layer: "+this->label+".");
            }
            if(this->input.size() != input.size())
                throw runtime_error("Input size does not match expected input size for layer: "+this->label+".");

            this->input = input;
            vector<double> op = vector<double>(weights.size(), 0);
            try{
                for(int i=0 ; i < weights.size() ; i++){
                    for(int j=0 ; j < weights[i].size() ; j++){
                        op[i] += (input[j] * weights[i][j].val);
                    }
                }

                // cout<<this->label<<":[ ";
                // for(double d: op)
                //     cout<<d<<", ";
                // cout<<"\b\b ]\n";

                return op;
            }
            catch(string s){
                throw runtime_error("Error in forwarding from dense layer: '"+this->label+"'.");
            }
        }

        vector<double> back_prop(vector<double> grads){
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads for layer: "+this->label+".");
            if(grads.size() != weights.size())
                throw runtime_error("Input grads size does not match no. of neurons for layer: "+this->label+".");

            vector<double> op = vector<double>(weights[0].size(), 0);
            for(int i=0 ; i<weights.size() ; i++){
                for(int j=0 ; j<weights[i].size() ; j++){
                    weights[i][j].grad += input[j] * grads[i];
                    op[j] += weights[i][j].val * grads[i];
                }
            }

            // cout<<this->label<<":[ ";
            // for(double d: op)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

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

            // cout<<this->label<<" output:[ ";
            // for(double d: op)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

            return op;
        }

        vector<double> back_prop(vector<double> grads){
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads for layer: "+this->label+".");
            if(grads.size() != input.size())
                throw runtime_error("grads size does not match no. of inputs for layer: "+this->label+".");

            // vector<double> op = vector<double>(grads.size(), 0);
            // double exps[input.size()];
            // double sum=0;
            // for(int i=0 ; i<input.size() ; i++){
            //     exps[i] = exp(input[i]);
            //     sum += exps[i];
            // }
            // for(int i=0 ; i<input.size() ; i++){
            //     for(int j=0 ; j<input.size() ; j++){
            //         op[i] += grads[i] * (exps[i]/sum) * ((i==j ? 1 : 0)-exps[j]/sum);
            //     }
            // }

            // // cout<<this->label<<" grads:[ ";
            // // for(double d: op)
            // //     cout<<d<<", ";
            // // cout<<"\b\b ]\n";

            // return op;

            return grads;
        }

        void update_weights(double lr = 0.1){}
};

class ReLU : public Layer{
    public:
        vector<double> input;

        ReLU(){}

        ReLU(string label, int size){
            this->label = label;
            this->input = vector<double>(size);
        }

        vector<double> forward(vector<double> input){
            if(input.size() != this->input.size()){
                throw runtime_error("Invalid input size for softmax layer: "+this->label);
            }
            this->input = input;
            for(int i=0 ; i<input.size() ; i++)
                if(input[i] < 0)
                    input[i] = 0;

            // cout<<this->label<<":[ ";
            // for(double d: input)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

            return input;
        }

        vector<double> back_prop(vector<double> grads){
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads for layer: "+this->label+".");
            if(grads.size() != input.size())
                throw runtime_error("grads size does not match no. of inputs for layer: "+this->label+".");

            for(int i = 0 ; i < grads.size() ; i++)
                if(input[i] <= 0)
                    grads[i] = 0;
            
            // cout<<this->label<<":[ ";
            // for(double d: grads)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

            return grads;
        }

        void update_weights(double lr = 0.1){}
};