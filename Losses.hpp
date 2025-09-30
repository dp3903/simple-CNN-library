#include <vector>
#include <cmath>
#include <stdexcept>
using namespace std;

class LossFunction{
    public:
        virtual double calculate(vector<double> actual, vector<double> expected) = 0;
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
};