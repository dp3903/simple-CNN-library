#include <vector>
#include <cmath>
#include <stdexcept>
using namespace std;

class LossFunction{
    public:
        virtual double calculate(vector<double> actual, vector<double> expected){
            throw runtime_error("Invalid input for Loss function.");
        };
        virtual double calculate(vector<vector<double>> actual, vector<vector<double>> expected){
            throw runtime_error("Invalid input for Loss function.");
        };
        virtual double calculate(vector<vector<vector<double>>> actual, vector<vector<vector<double>>> expected){
            throw runtime_error("Invalid input for Loss function.");
        };
};


class MSELoss : public LossFunction{
    public:
        double calculate(vector<double> actual, vector<double> expected){
            if(actual.size() != expected.size())
                throw runtime_error("Expected output size does not match model output size.");

            double loss = 0.0;
            for(int i=0 ; i<actual.size() ; i++){
                loss += pow(actual[i]-expected[i], 2);
            }
            return loss;
        }
        
        double calculate(vector<vector<double>> actual, vector<vector<double>> expected){
            if(actual.size() != expected.size())
                throw runtime_error("Expected output size does not match model output size.");

            double loss = 0.0;
            for(int i=0 ; i<actual.size() ; i++){
                loss += this->calculate(actual[i], expected[i]);
            }
            return loss;
        }
        
        double calculate(vector<vector<vector<double>>> actual, vector<vector<vector<double>>> expected){
            if(actual.size() != expected.size())
                throw runtime_error("Expected output size does not match model output size.");

            double loss = 0.0;
            for(int i=0 ; i<actual.size() ; i++){
                loss += this->calculate(actual[i], expected[i]);
            }
            return loss;
        }
};