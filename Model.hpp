#include "Layer.hpp"
#include <memory>

class Model{
    public:
        string name;
        vector<unique_ptr<Layer>> layers;
        
        Model(){}

        // constructor taking an initializer_list of unique_ptrs
        Model(std::initializer_list<std::unique_ptr<Layer>> list) {
            for (auto& l : list) {
                layers.push_back(std::move(const_cast<std::unique_ptr<Layer>&>(l)));
            }
        }

        Model(string name, std::initializer_list<std::unique_ptr<Layer>> list){
            this->name = name;
            for (auto& l : list) {
                layers.push_back(std::move(const_cast<std::unique_ptr<Layer>&>(l)));
            }
        }
};