#include "Layers.hpp"
#include "Losses.hpp"
#include <memory>
#include <iomanip>
#include <iostream>


class Model{
    public:
        string name;
        vector<Layer*> layers;
        LossFunction* loss;
        bool trainable = true;
        
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

        Tensor run(Tensor ip){
            for(Layer* l: layers){
                try{
                    ip = l->forward(ip);
                }
                catch (const exception& e) {
                    std::cerr << "Error: " << e.what() << std::endl;
                    exit(1);
                }
            }
            return ip;
        }
        
        Tensor back(Tensor grads){
            for(int u=layers.size()-1 ; u>=0 ; u--){
                try{
                    grads = layers[u]->back_prop(grads);
                }
                catch (const exception& e) {
                    std::cerr << "Error: " << e.what() << std::endl;
                    exit(1);
                }
            }
            return grads;
        }

        vector<double> train(Batch batch, int iterations=1, double learning_rate=0.1){
            vector<double> loses = vector<double>(iterations);
            // cout<<"\n===============Batch Training Starting===============\n";
            for(int i=0 ; i < iterations ; i++){
                double iteration_loss = 0.0;
                for(pair<Tensor,Tensor> sample : batch){
                    Tensor op = this->run(sample.first);
                
                    iteration_loss += this->loss->calculate(op,sample.second);
                    Tensor op_grads = (op - sample.second) / size(op);
                    this->back(op_grads);

                    
                    // cout<<setw(20)<<"Predicted output: [ ";
                    // for(double d: op)
                    //     cout<<d<<", ";
                    // cout<<"\b\b ]\n";
                    // cout<<setw(20)<<"Actual output: [ ";
                    // for(double d: sample.second)
                    //     cout<<d<<", ";
                    // cout<<"\b\b ]\n";
                }

                if(this->trainable){
                    for(Layer* l: layers)
                       l->update_weights(learning_rate);
                }

                loses[i] = iteration_loss/batch.size();
                // cout<<"Iteration: "<<i<<" Loss: "<<loses[i]<<'\r';
            }
            // cout<<"\n===============Batch Training Complete===============\n";
            return loses;
        }
};