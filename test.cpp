#include "CNN.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
using namespace std;
#define Dataset vector<pair<vector<double>,vector<double>>>
#define Data pair<vector<double>,vector<double>>

// Reads a batch of MNIST samples from CSV
// Returns vector of pairs: (input vector, one-hot label vector)
Dataset load_mnist_batch(const string& filename, size_t batch_size, size_t start_line = 0) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file " + filename);
    }
    // Skip header line
    string header;
    getline(file, header);

    vector<pair<vector<double>,vector<double>>> batch;
    batch.reserve(batch_size);

    string line;
    size_t line_count = 0;

    // Skip lines until start_line
    while (line_count < start_line && getline(file, line)) {
        line_count++;
    }

    // Now read batch_size lines
    size_t loaded = 0;
    while (loaded < batch_size && getline(file, line)) {
        stringstream ss(line);
        string cell;

        // Read label
        getline(ss, cell, ',');
        int label = stoi(cell);

        // Build one-hot encoded vector
        vector<double> output(10, 0.0);
        output[label] = 1.0;

        // Read pixels
        vector<double> input;
        input.reserve(784);
        while (getline(ss, cell, ',')) {
            double pixel = stod(cell) / 255.0; // normalize 0-1
            input.push_back(pixel);
        }

        batch.push_back({input, output});
        loaded++;
    }

    return batch;
}


// each training example: (input_vector, label_vector)
// vector< pair< vector<double>, vector<double> > > dataset = {
//     {{0.12, 0.55, 0.33, 0.78, 0.90, 0.14, 0.66, 0.42, 0.81, 0.27}, {1, 0, 0}},
//     {{0.99, 0.05, 0.47, 0.31, 0.22, 0.63, 0.15, 0.82, 0.74, 0.40}, {0, 1, 0}},
//     {{0.34, 0.71, 0.59, 0.20, 0.88, 0.41, 0.13, 0.93, 0.56, 0.77}, {0, 0, 1}},
//     {{0.25, 0.36, 0.91, 0.44, 0.58, 0.07, 0.62, 0.83, 0.19, 0.52}, {1, 0, 0}},
//     {{0.80, 0.68, 0.29, 0.55, 0.17, 0.33, 0.71, 0.09, 0.47, 0.61}, {0, 1, 0}},
//     {{0.06, 0.88, 0.23, 0.39, 0.50, 0.77, 0.84, 0.28, 0.62, 0.19}, {0, 0, 1}},
//     {{0.45, 0.11, 0.95, 0.70, 0.08, 0.53, 0.22, 0.41, 0.64, 0.87}, {1, 0, 0}},
//     {{0.73, 0.12, 0.38, 0.49, 0.21, 0.96, 0.65, 0.34, 0.59, 0.07}, {0, 1, 0}},
//     {{0.19, 0.27, 0.54, 0.62, 0.85, 0.43, 0.31, 0.08, 0.97, 0.75}, {0, 0, 1}},
//     {{0.92, 0.44, 0.36, 0.51, 0.26, 0.67, 0.15, 0.79, 0.33, 0.58}, {1, 0, 0}}
// };

int main(){
    
    Model test_model = Model({
        new Dense("Layer1", 784, 256),
        new ReLU("relu1",256),
        new Dense("Layer2", 256, 128),
        new ReLU("relu2",128),
        new Dense("Layer3", 128, 10),
        new Softmax("softmax1", 10)
    });
    
    int n_epochs = 5;
    int n_iterations = 1;
    int batch_size = 1;
    int dataset_size = 10;
    string file_path = "./archive/mnist_train.csv";

    for(int epoch=0 ; epoch<n_epochs ; epoch++){
        cout<<"Epoch: "<<epoch<<endl;
            int start_line=0;
            vector<double> loses;
            while(start_line < dataset_size){
                Dataset batch;
                if(start_line + batch_size > dataset_size){
                    batch = load_mnist_batch(file_path, dataset_size-start_line, start_line);
                }
                else{
                    batch = load_mnist_batch(file_path, batch_size, start_line);
                }
                vector<double> l = test_model.train(
                    batch, // batch
                    100,     // iterations
                    0.001       // learning rate
                );
                loses.insert(loses.end(),l.begin(),l.end());
                cout<<"loss: "<<l.back()<<"\r";
                start_line += batch_size;
            }
            cout<<"Epoch: "<<epoch<<" loss: "<<loses.back()<<endl;
    }

    cout<<"=====Testing with random input=====\n";
    for(int i=0 ; i<5 ; i++){
        int x = rand()%dataset_size;
        Data data = load_mnist_batch(file_path, 1, x)[0];
        vector<double> op = test_model.run(data.first);
        cout<<setw(20)<<"Predicted output: [ ";
        for(double d: op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: data.second)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
    }
}