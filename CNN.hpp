#include "Layers.hpp"
#include "Losses.hpp"
#include <memory>
#include <iostream>

class Model{
    public:
        string name;
        vector<Layer*> layers;
        LossFunction* loss;
        
        Model(){}

        // constructor taking an initializer_list of unique_ptrs
        Model(vector<Layer*> list, LossFunction* loss = new MSELoss()){
            layers = list;
            this->loss = loss;
        }

        Model(string name, vector<Layer*> list, LossFunction* loss = new MSELoss()){
            this->name = name;
            layers = list;
            this->loss = loss;
        }

        vector<double> run(vector<double> ip){
            vector<Value> op;
            for(double d: ip){
                op.push_back(Value("input",d));
            }
            for(Layer* l: layers){
                try{
                    op = l->forward(op);
                }
                catch (const exception& e) {
                    std::cerr << "Error: " << e.what() << std::endl;
                    exit(1);
                }
            }
            vector<double> ans;
            for(Value v: op)
                ans.push_back(v.val);
            return ans;
        }

        vector<double> train(vector<pair<vector<double>,vector<double>>> dataset, int iterations=1){
            vector<double> loses = vector<double>(iterations);
            cout<<"\n===============Training Starting===============\n";
            for(int i=0 ; i < iterations ; i++){
                double iteration_loss = 0.0;
                for(pair<vector<double>,vector<double>> sample : dataset){
                    vector<double> op = this->run(sample.first);
                
                    try{
                        iteration_loss += this->loss->calculate(op,sample.second);
                    }
                    catch (const exception& e) {
                        std::cerr << "Error: " << e.what() << std::endl;
                        exit(1);
                    }
                }
                loses[i] = iteration_loss;
                cout<<"Iteration: "<<i<<" Loss: "<<iteration_loss<<'\r';
            }
            cout<<"\n===============Training Complete===============\n";
            return loses;
        }
};