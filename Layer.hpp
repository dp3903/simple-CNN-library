#include <ostream>
#include <cmath>
#include <vector>
#include "Value.hpp"

class Layer{
    public:
        string label;

        Layer(){}
        Layer(string name){
            this->label = name;
        }

        vector<Value> forward(vector<double> input){
            throw "Invalid input.";
        }

        vector<Value> forward(vector<Value> input){
            throw "Invalid input.";
        }
        
        vector<vector<Value>> forward(vector<vector<double>> input){
            throw "Invalid input.";
        }

        vector<vector<Value>> forward(vector<vector<Value>> input){
            throw "Invalid input.";
        }
};