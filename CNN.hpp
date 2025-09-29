#include "Dense.hpp"
#include <memory>

class Model{
    public:
        string name;
        vector<Layer*> layers;
        
        Model(){}

        // constructor taking an initializer_list of unique_ptrs
        Model(vector<Layer*> list) {
            layers = list;
        }

        Model(string name, vector<Layer*> list){
            this->name = name;
            layers = list;
        }
};