#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
using namespace std;


class LossFunction{
    public:
        virtual double calculate(Tensor actual, Tensor expected){
            throw runtime_error("Invalid input for Loss function.");
        };
};


class MSELoss : public LossFunction{
    public:
        double calculate(Tensor t_actual, Tensor t_expected){
            if(holds_alternative<Tensor1D>(t_actual) && holds_alternative<Tensor1D>(t_expected)){
                const Tensor1D& actual = std::get<Tensor1D>(t_actual);
                const Tensor1D& expected = std::get<Tensor1D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                for(int i=0 ; i<actual.size() ; i++){
                    loss += pow(actual[i]-expected[i], 2);
                }
                return loss;
            }
            if(holds_alternative<Tensor2D>(t_actual) && holds_alternative<Tensor2D>(t_expected)){
                const Tensor2D& actual = std::get<Tensor2D>(t_actual);
                const Tensor2D& expected = std::get<Tensor2D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                for(int i=0 ; i<actual.size() ; i++){
                    loss += this->calculate(actual[i], expected[i]);
                }
                return loss;
            }
            if(holds_alternative<Tensor3D>(t_actual) && holds_alternative<Tensor3D>(t_expected)){
                const Tensor3D& actual = std::get<Tensor3D>(t_actual);
                const Tensor3D& expected = std::get<Tensor3D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                for(int i=0 ; i<actual.size() ; i++){
                    loss += this->calculate(actual[i], expected[i]);
                }
                return loss;
            }
            throw invalid_argument("Invalid arguments to MSELoss function.");
        }
};