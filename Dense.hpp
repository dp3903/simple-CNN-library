#include <ostream>
#include <cmath>
#include <vector>
#include "Layer.hpp"
using namespace std;

class Dense : public Layer{
    public:
        vector<vector<double>> weights;
        vector<Value> values;

        Dense(){}

        Dense(string label, int input_size, int output_size) : Layer(label){
            if(weights.size() != values.size()){
                throw "Size of weights vector must match no. of values.";
            }
            this->label = (label.size()!=0 ? label : "temp");
            this->weights = vector<vector<double>>(output_size, vector<double>(input_size));
            this->values = vector<Value>(output_size);
            for(int i=0 ; i < weights.size() ; i++){
                for(int j=0 ; j < weights[i].size() ; j++){
                    weights[i][j] = double(rand()) / RAND_MAX;
                }
            }
        }

        vector<Value> forward(vector<double> input){
            if(!weights.size() || !values.size()){
                throw "Cannot forword on empty layer.";
            }

            vector<Value> output = vector<Value>(weights.size());
            try{
                for(int i=0 ; i < weights.size() ; i++){
                    for(int j=0 ; j < weights[i].size() ; j++){
                        output[i] = output[i] + (Value("",input[j]) * Value("",weights[i][j]));
                    }
                }
                this->values = output;
                return output;
            }
            catch(string s){
                throw "Error in forwarding from dense layer: '"+this->label+"'.";
            }
        }
        
        vector<Value> forward(vector<Value> input){
            if(!weights.size() || !values.size()){
                throw "Cannot forword on empty layer.";
            }

            vector<Value> output = vector<Value>(weights.size());
            try{
                for(int i=0 ; i < weights.size() ; i++){
                    output.push_back(Value());
                    for(int j=0 ; j < weights[i].size() ; j++){
                        output[i] = output[i] + (input[j]*Value("",weights[i][j]));
                    }
                }
                this->values = output;
                return output;
            }
            catch(string s){
                throw "Error in forwarding from dense layer: '"+this->label+"'.";
            }
        }
};