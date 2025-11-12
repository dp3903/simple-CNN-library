#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
using namespace std;


class LossFunction{
    public:
        virtual pair<double, Tensor> calculate(Tensor actual, Tensor expected){
            throw runtime_error("Invalid input for Loss function.");
        };
};


class MSELoss : public LossFunction{
    public:
        pair<double, Tensor> calculate(Tensor t_actual, Tensor t_expected) override{
            if(holds_alternative<Tensor1D>(t_actual) && holds_alternative<Tensor1D>(t_expected)){
                const Tensor1D& actual = std::get<Tensor1D>(t_actual);
                const Tensor1D& expected = std::get<Tensor1D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                Tensor1D grads = Tensor1D(actual.size());
                double loss = 0.0;
                for(int i=0 ; i<actual.size() ; i++){
                    double x = actual[i] - expected[i];
                    grads[i] = x;
                    loss += x*x;
                }
                loss /= 2;
                return {loss, grads};
            }
            if(holds_alternative<Tensor2D>(t_actual) && holds_alternative<Tensor2D>(t_expected)){
                const Tensor2D& actual = std::get<Tensor2D>(t_actual);
                const Tensor2D& expected = std::get<Tensor2D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                Tensor2D grads = Tensor2D(actual.size());
                for(int i=0 ; i<actual.size() ; i++){
                    auto [r_loss, r_grads] = this->calculate(actual[i], expected[i]);
                    grads[i] = std::get<Tensor1D>(r_grads);
                    loss += r_loss;
                }
                return {loss, grads};
            }
            if(holds_alternative<Tensor3D>(t_actual) && holds_alternative<Tensor3D>(t_expected)){
                const Tensor3D& actual = std::get<Tensor3D>(t_actual);
                const Tensor3D& expected = std::get<Tensor3D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                Tensor3D grads = Tensor3D(actual.size());
                for(int i=0 ; i<actual.size() ; i++){
                    auto [r_loss, r_grads] = this->calculate(actual[i], expected[i]);
                    grads[i] = std::get<Tensor2D>(r_grads);
                    loss += r_loss;
                }
                return {loss, grads};
            }
            throw invalid_argument("Invalid arguments to MSELoss function.");
        }
};

class BinaryCrossEntropyLoss : public LossFunction{
    public:
        pair<double, Tensor> calculate(Tensor t_actual, Tensor t_expected) override{
            if(holds_alternative<Tensor1D>(t_actual) && holds_alternative<Tensor1D>(t_expected)){
                const Tensor1D& actual = std::get<Tensor1D>(t_actual);
                const Tensor1D& expected = std::get<Tensor1D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                Tensor1D grads = Tensor1D(actual.size());
                double loss = 0.0;
                for(int i=0 ; i<actual.size() ; i++){
                    loss += -(expected[i]*log(actual[i]) + (1-expected[i])*log(1-actual[i]));
                    grads[i] = (actual[i] - expected[i]) / (actual[i] * (1-actual[i]));
                }
                return {loss, grads};
            }
            if(holds_alternative<Tensor2D>(t_actual) && holds_alternative<Tensor2D>(t_expected)){
                const Tensor2D& actual = std::get<Tensor2D>(t_actual);
                const Tensor2D& expected = std::get<Tensor2D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                Tensor2D grads = Tensor2D(actual.size());
                for(int i=0 ; i<actual.size() ; i++){
                    auto [r_loss, r_grads] = this->calculate(actual[i], expected[i]);
                    grads[i] = std::get<Tensor1D>(r_grads);
                    loss += r_loss;
                }
                return {loss, grads};
            }
            if(holds_alternative<Tensor3D>(t_actual) && holds_alternative<Tensor3D>(t_expected)){
                const Tensor3D& actual = std::get<Tensor3D>(t_actual);
                const Tensor3D& expected = std::get<Tensor3D>(t_expected);
                if(actual.size() != expected.size())
                    throw runtime_error("Expected output size does not match model output size.");
                
                double loss = 0.0;
                Tensor3D grads = Tensor3D(actual.size());
                for(int i=0 ; i<actual.size() ; i++){
                    auto [r_loss, r_grads] = this->calculate(actual[i], expected[i]);
                    grads[i] = std::get<Tensor2D>(r_grads);
                    loss += r_loss;
                }
                return {loss, grads};
            }
            throw invalid_argument("Invalid arguments to MSELoss function.");
        }
};