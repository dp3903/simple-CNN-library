#include "CNN.hpp"
using namespace std;

int main(){
    Model test_model = Model({
        new Dense("Layer1", 128, 64),
        new Dense("Layer2", 64, 10)
    });
}