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

                for(Layer* l: layers)
                    l->update_weights(learning_rate);
                
                loses[i] = iteration_loss/batch.size();
                // cout<<"Iteration: "<<i<<" Loss: "<<loses[i]<<'\r';
            }
            // cout<<"\n===============Batch Training Complete===============\n";
            return loses;
        }

        void summary(){
            int total_params = 0, trainable_params = 0;
            cout<<endl;
            cout<<"\t-"<<(string("-")*131)<<endl;
            cout<<"\t|"<<setw(25)<<"Layer"<<" "<<setw(25)<<"input_shape"<<" "<<setw(25)<<"output_shape"<<" "<<setw(25)<<"Parameters"<<" "<<setw(25)<<"Trainable"<<" |"<<endl;
            cout<<"\t-"<<(string("-")*131)<<endl;
            for(Layer* l: this->layers){
                int params = l->parameter_count();
                cout<<"\t|"<<setw(25)<<l->label<<" "<<setw(25)<<l->input_shape<<" "<<setw(25)<<l->output_shape<<" "<<setw(25)<<(params>=0?to_string(params):"uninitialized")<<" "<<setw(25)<<(l->trainable?"true":"false")<<" |"<<endl;
                if(total_params >= 0 && params >= 0){
                    total_params += params;
                    if(l->trainable)
                        trainable_params += params;
                }
                else{
                    total_params = -1;
                    trainable_params = -1;
                }
            }
            cout<<"\t-"<<(string("-")*131)<<endl;
            cout<<"\t|"<<setw(25)<<"Total layers"<<" "<<setw(25)<<(this->layers.size())<<" |"<<endl;
            cout<<"\t|"<<setw(25)<<"Total parameters"<<" "<<setw(25)<<(total_params>=0?to_string(total_params):"uninitialized")<<" |"<<endl;
            cout<<"\t|"<<setw(25)<<"Trainable parameters"<<" "<<setw(25)<<(trainable_params>=0?to_string(trainable_params):"uninitialized")<<" |"<<endl;
            cout<<"\t|"<<setw(25)<<"Non-Trainable parameters"<<" "<<setw(25)<<(trainable_params>=0?to_string(total_params - trainable_params):"uninitialized")<<" |"<<endl;
            cout<<"\t-"<<(string("-")*53)<<endl;
            cout<<endl;
        }

        void set_traianable(bool s){
            for(Layer* l: layers)
                l->trainable = false;
        }

        bool get_traianable(){
            for(Layer* l: layers)
                if(l->trainable)
                    return true;
            return false;
        }
};


vector<Layer*> AutoInitializer(vector<Layer*> layers){
    for(int i=0 ; i<layers.size() ; i++){
        // const std::type_info& info = typeid(*layers[i]);
        // cout<<"Layer: "<<i<<" name: "<<info.name()<<endl;
        
    }
    return {};
}