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
        
        virtual vector<vector<vector<double>>> forward(vector<vector<vector<double>>> input){
            throw runtime_error("Invalid input for layer: "+this->label+".");
        }
        
        virtual vector<double> back_prop(vector<double> grads){
            throw runtime_error("Invalid gradients for layer: "+this->label+".");
        }

        virtual vector<vector<double>> back_prop(vector<vector<double>> grads){
            throw runtime_error("Invalid gradients for layer: "+this->label+".");
        }

        virtual vector<vector<vector<double>>> back_prop(vector<vector<vector<double>>> grads){
            throw runtime_error("Invalid gradients for layer: "+this->label+".");
        }
        
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

class Conv2d : public Layer{
    public:
        vector<vector<vector<vector<Value>>>> filters;
        vector<vector<vector<double>>> input;
        int stride;
        int padding;
        int filter_size;
        tuple<int,int,int> input_dim;
        tuple<int,int,int> output_dim;

        Conv2d(){}

        Conv2d(string label, tuple<int,int,int> input_dim, int filter_size, int no_of_filters, int stride=1, int padding=0){
            if(stride <= 0)
                throw runtime_error("Stride cannot be 0 or negative for Conv2d layer: "+label);
            if(padding < 0)
                throw runtime_error("Padding cannot be negative for Conv2d layer: "+label);
            if(get<0>(input_dim)<=0 || get<1>(input_dim)<=0 || get<2>(input_dim)<=0)
                throw runtime_error("Input dimension cannot be less than 0 for Conv2d layer: "+label);

            this->label = label;
            this->input_dim = input_dim;
            this->output_dim = {
                ((get<0>(input_dim) + 2*padding - filter_size) / stride + 1),
                ((get<1>(input_dim) + 2*padding - filter_size) / stride + 1),
                no_of_filters
            };
            this->filters = vector<vector<vector<vector<Value>>>>(no_of_filters, vector<vector<vector<Value>>>(get<2>(input_dim), vector<vector<Value>>(filter_size, vector<Value>(filter_size))));
            this->input = vector<vector<vector<double>>>(get<2>(input_dim), vector<vector<double>>(get<0>(input_dim), vector<double>(get<1>(input_dim))));
            this->stride = stride;
            this->filter_size = filter_size;
            this->padding = padding;
            for(int i=0 ; i < filters.size() ; i++){
                for(int j=0 ; j < filters[i].size() ; j++){
                    for(int k=0 ; k < filters[i][j].size() ; k++){
                        for(int l=0 ; l < filters[i][j][k].size() ; l++){
                            filters[i][j][k][l] = Value(
                                "Conv2d: "+this->label+" param("+to_string(i)+','+to_string(j)+','+to_string(k)+','+to_string(l)+").",
                                generate_random_in_range(-0.1, 0.1)
                            );
                        }
                    }
                }
            }
        }

        vector<vector<vector<double>>> forward(vector<vector<vector<double>>> input){
            if(input.size() != get<2>(input_dim))
                throw runtime_error("Invalid input dimensions for layer: "+this->label);
            
            vector<vector<vector<double>>> ans = vector<vector<vector<double>>>(get<2>(output_dim), vector<vector<double>>(get<0>(output_dim), vector<double>(get<1>(output_dim),0)));
            for (int oc = 0; oc < get<2>(output_dim); oc++) {
                for (int out_x = 0; out_x < get<0>(output_dim); out_x++) {
                    for (int out_y = 0; out_y < get<1>(output_dim); out_y++) {
                        double sum = 0.0;
                        for (int ic = 0; ic < get<2>(input_dim); ic++) {
                            for (int i = 0; i < filter_size; i++) {
                                for (int j = 0; j < filter_size; j++) {
                                    int in_x = out_x * stride + i - padding;
                                    int in_y = out_y * stride + j - padding;
                                    if (in_x >= 0 && in_x < get<0>(input_dim) &&
                                        in_y >= 0 && in_y < get<1>(input_dim)) {
                                        sum += filters[oc][ic][i][j].val * input[ic][in_x][in_y];
                                    }
                                }
                            }
                        }
                        ans[oc][out_x][out_y] = sum;
                    }
                }
            }

            this -> input = input;

            return ans;
        }
        
        vector<vector<vector<double>>> back_prop(vector<vector<vector<double>>> grads){
            vector<vector<vector<double>>> ans;

            return ans;
        }

        void update_weights(double learning_rate = 0.1){
            for(int i=0 ; i < filters.size() ; i++){
                for(int j=0 ; j < filters[i].size() ; j++){
                    for(int k=0 ; k < filters[i][j].size() ; k++){
                        for(int l=0 ; l < filters[i][j][k].size() ; l++){
                            filters[i][j][k][l].val -= learning_rate * filters[i][j][k][l].grad;
                            filters[i][j][k][l].grad = 0;
                        }
                    }
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