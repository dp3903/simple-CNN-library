#include "CNN.hpp"
using namespace std;

int main(){
    Model test_model = Model({
        DenseLayer("Layer1", 128, 64),
        DenseLayer("Layer2", 64, 10)
    });
}