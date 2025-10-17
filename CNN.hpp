#include "Layers.hpp"
#include "Losses.hpp"
#include <memory>
#include <iomanip>
#include <iostream>

template <typename T>
std::enable_if_t<
    (std::is_same_v<T, Tensor1D> || 
    std::is_same_v<T, Tensor2D> || 
    std::is_same_v<T, Tensor3D>),
    int
>
Tensor_size(T tensor){
    if constexpr (std::is_same_v<T, Tensor1D>)
        return tensor.size();
    else if constexpr (std::is_same_v<T, Tensor2D>)
        return (tensor.size() * tensor[0].size());
    else
        return (tensor.size() * tensor[0].size() * tensor[0][0].size());
}

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

        variant<Tensor1D,Tensor2D,Tensor3D> run(variant<Tensor1D,Tensor2D,Tensor3D> ip){
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
        
        variant<Tensor1D,Tensor2D,Tensor3D> back(variant<Tensor1D,Tensor2D,Tensor3D> grads){
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

        template <typename T, typename V>
        std::enable_if_t<
            ((
            std::is_same_v<T, Tensor1D> || 
            std::is_same_v<T, Tensor2D> || 
            std::is_same_v<T, Tensor3D>)
            &&
            (std::is_same_v<V, Tensor1D> || 
            std::is_same_v<V, Tensor2D> || 
            std::is_same_v<V, Tensor3D>)),
            vector<double> 
        >
        train(vector<pair<T,V>> batch, int iterations=1, double learning_rate=0.1){
            vector<double> loses = vector<double>(iterations);
            // cout<<"\n===============Batch Training Starting===============\n";
            for(int i=0 ; i < iterations ; i++){
                double iteration_loss = 0.0;
                for(pair<T,V> sample : batch){
                    V op = std::get<V>(this->run(sample.first));
                
                    iteration_loss += this->loss->calculate(op,sample.second);
                    V op_grads = (op - sample.second) / Tensor_size(op);
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