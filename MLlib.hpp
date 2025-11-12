#pragma once
#include "Layers.hpp"
#include "Losses.hpp"
#include "Initializer.hpp"
#include "Optimizers.hpp"
#include <memory>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "cereal/archives/binary.hpp"
#include "cereal/types/memory.hpp"

class Model{
    public:
        string name;
        vector<unique_ptr<Layer>> layers;
        unique_ptr<LossFunction> loss;

        Model() = default; // Use default for empty constructor

        Model(
            vector<unique_ptr<Layer>> layer_list,
            unique_ptr<LossFunction> loss_func = make_unique<MSELoss>()
        ) : layers(move(layer_list)), loss(move(loss_func))
        {}

        Model(
            string model_name,
            vector<unique_ptr<Layer>> layer_list,
            unique_ptr<LossFunction> loss_func = make_unique<MSELoss>()
        ) : name(move(model_name)), layers(move(layer_list)), loss(move(loss_func))
        {}

        Model(
            initializer_list<Layer*> list,
            unique_ptr<LossFunction> loss_func = make_unique<MSELoss>()
        ) : loss(move(loss_func))
        {
            layers.reserve(list.size());
            for (Layer* ptr : list) {
                // For each raw pointer, create a unique_ptr that takes ownership
                // and add it to our member vector.
                layers.emplace_back(ptr);
            }
        }

        Model(
            string model_name,
            initializer_list<Layer*> list,
            unique_ptr<LossFunction> loss_func = make_unique<MSELoss>()
        ) : name(move(model_name)), loss(move(loss_func))
        {
            layers.reserve(list.size());
            for (Layer* ptr : list) {
                layers.emplace_back(ptr);
            }
        }

        ~Model() = default;

        Tensor run(Tensor ip){
            for(const auto& l: layers){
                try{
                    ip = l->forward(ip);
                }
                catch (const exception& e) {
                    cerr << "Error: " << e.what() << endl;
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
                    cerr << "Error: " << e.what() << endl;
                    exit(1);
                }
            }
            return grads;
        }

        vector<double> train(Batch batch, int iterations=1){
            vector<double> loses = vector<double>(iterations);
            // cout<<"\n===============Batch Training Starting===============\n";
            for(int i=0 ; i < iterations ; i++){
                double iteration_loss = 0.0;
                for(pair<Tensor,Tensor> sample : batch){
                    Tensor op = this->run(sample.first);
                
                    auto [loss, op_grads] = this->loss->calculate(op,sample.second);
                    iteration_loss += loss;
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

                // optim->step();
                
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
            for(const auto& l: this->layers){
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
            for(const auto& l: layers)
                l->trainable = false;
        }

        bool get_traianable(){
            for(const auto& l: layers)
                if(l->trainable)
                    return true;
            return false;
        }

        template <class Archive>
        void serialize(Archive& archive) {
            archive(
                CEREAL_NVP(name),      // Saves the label
                CEREAL_NVP(layers)
            );
        }
};


// --- SAVING ---
void save_model(const Model& model, const string& path) {
    ofstream os(path, ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(model); // That's it! Cereal saves the whole polymorphic vector.
}

// --- LOADING ---
void load_model(Model& model, const string& path) {
    ifstream is(path, ios::binary);
    cereal::BinaryInputArchive archive(is);
    archive(model); // Cereal reconstructs the entire model.
}