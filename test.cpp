#define NOMINMAX
#define NODATA
#include "indicators.hpp"
#include "CNN.hpp"
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;


// Reads a batch of MNIST samples from CSV
// Returns vector of pairs: (input vector, one-hot label vector)
Batch load_mnist_batch(const string& filename, size_t batch_size, size_t start_line = 0) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file " + filename);
    }
    // Skip header line
    if(start_line == 0){
        string header;
        getline(file, header);
    }

    Batch batch;
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

Batch reshape_mnist_batch(Batch batch){
    Batch output;
    int height = 28, width = 28, channels = 1;
    Flatten fl;
    fl.input_shape = {28,28,1}; // we use reverse flatten i.e. backprop of flatten to unflatten the input.
    for(Data d: batch){
        pair<Tensor,Tensor> temp;
        temp.second = d.second;
        temp.first = fl.back_prop(d.first);
        output.push_back(temp);
    }
    return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_ann(){
    
    Model test_model = Model({
        new Dense("Layer1", 784, 256),
        new ReLU("relu1"),
        new Dense("Layer2", 256, 128),
        new ReLU("relu2"),
        new Dense("Layer3", 128, 10),
        new Softmax("softmax1")
    });
    
    int n_epochs = 5;
    int n_iterations_per_batch = 1;
    int batch_size = 10;
    int dataset_size = 1000;
    double learning_rate = 0.01;
    // Calculate the total number of batches for 1 epoch
    int total_batches = (dataset_size + batch_size - 1) / batch_size;
    string file_path = "./archive/mnist_train.csv";

    cout<<"\n=====Testing with random input before training=====\n";
    for(int i=0 ; i<5 ; i++){
        int x = rand()%dataset_size;
        Data data = load_mnist_batch(file_path, 1, x)[0];
        Tensor1D op = std::get<Tensor1D>(test_model.run(data.first));
        cout<<setw(20)<<"Predicted output: [ ";
        for(double d: op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
        Tensor1D actual_op = std::get<Tensor1D>(data.second);
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: actual_op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
    }

    indicators::show_console_cursor(false);
    indicators::ProgressBar p{
        indicators::option::BarWidth{20},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::End{"]"},
        indicators::option::ForegroundColor{indicators::Color::yellow},
        indicators::option::PrefixText{"Epoch: 0"},
        indicators::option::PostfixText{"Batch: 0 Loss: inf"},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::MaxProgress{static_cast<size_t>(total_batches * n_epochs)}
    };
    for(int epoch=0 ; epoch<n_epochs ; epoch++){
        p.set_option(indicators::option::PrefixText{"Epoch: "+to_string(epoch)});
        int start_line=1, i=1;
        vector<double> loses;
        while(start_line < dataset_size){
            Batch batch;
            if(start_line + batch_size > dataset_size){
                batch = load_mnist_batch(file_path, dataset_size-start_line, start_line);
            }
            else{
                batch = load_mnist_batch(file_path, batch_size, start_line);
            }
            vector<double> l = test_model.train(
                batch,                          // batch
                n_iterations_per_batch,         // iterations
                learning_rate                   // learning rate
            );
            loses.insert(loses.end(),l.begin(),l.end());
            // cout<<"loss: "<<l.back()<<"\r";
            p.set_option(indicators::option::PostfixText{"Batch: "+to_string(i)+" Batch Loss: "+to_string(l.back())});
            p.tick();
            i++;
            start_line += batch_size;
        }
        cout<<endl;
    }
    indicators::show_console_cursor(true);
    std::cout << "\033[0m" << std::flush;  // reset colors to default


    cout<<"\n=====Testing with random input after training=====\n";
    for(int i=0 ; i<5 ; i++){
        int x = rand()%dataset_size;
        Data data = load_mnist_batch(file_path, 1, x)[0];
        Tensor1D op = std::get<Tensor1D>(test_model.run(data.first));
        cout<<setw(20)<<"Predicted output: [ ";
        for(double d: op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
        Tensor1D actual_op = std::get<Tensor1D>(data.second);
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: actual_op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_op(){

    vector<vector<double>> a = {
        {1,2,3},
        {4,5,6}
    };
    vector<vector<double>> b = {
        {1,2,3},
        {4,5,6}
    };
    cout<<a<<'\n'<<b<<'\n'<<(a+b)<<endl;     
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_convolution(){
    
    // Define input dimensions: (height, width, channels)
    tuple<int,int,int> input_dim = {4, 4, 1};

    // Create Conv2D layer
    Conv2D conv("TestConv", input_dim, 3, 1, 1, 0, 1); 
    // filter_size, no_of_filters, stride, padding, dilation
    cout<<get<0>(conv.output_dim)<<','<<get<1>(conv.output_dim)<<','<<get<2>(conv.output_dim)<<'\n';

    // Create simple input (1 channel of 4x4)
    Tensor3D input = Tensor3D(1, Tensor2D(get<0>(input_dim), Tensor1D(get<1>(input_dim))));

    // Fill with simple increasing pattern
    int val = 1;
    for (int i = 0; i < input[0].size(); i++) {
        for (int j = 0; j < input[0][i].size(); j++) {
            input[0][i][j] = val++;
        }
    }

    cout << "Input:\n";
    cout<<input[0];

    // Set filters to 1
    for (int oc = 0; oc < conv.filters.size(); oc++) {
        for (int ic = 0; ic < conv.filters[oc].size(); ic++){
            for(int x = 0; x < conv.filters[oc][ic].size(); x++){
                for(int y = 0; y < conv.filters[oc][ic][x].size(); y++){
                    conv.filters[oc][ic][x][y].val = 1;
                }
            }
        }
    }

    // Forward pass
    Tensor ip = input;
    auto output = std::get<Tensor3D>(conv.forward(ip));

    cout << "\nOutput dimensions: "
         << get<0>(conv.output_dim) << "x" << get<1>(conv.output_dim)
         << "x" << get<2>(conv.output_dim) << endl;

    cout << "\nOutput:\n";
    for (int oc = 0; oc < output.size(); oc++) {
        cout << "Channel " << oc << ":\n";
        cout << output[oc] << endl;
    }
    
    cout << "\nFilter grads before calc:\n";
    for (int oc = 0; oc < conv.filters.size(); oc++) {
        cout << "Output Channel " << oc << ":\n";
        for (int ic = 0; ic < conv.filters[oc].size(); ic++){
            cout << "\tInput Channel " << ic << ":\n";
            cout << conv.filters[oc][ic] << endl;
        }
    }
    
    Tensor grads = (Tensor3D){{
        {1,1},
        {1,1}
    }};
    vector<vector<vector<double>>> in_grads = std::get<Tensor3D>(conv.back_prop(grads));

    cout << "\nFilter grads after calc:\n";
    for (int oc = 0; oc < conv.filters.size(); oc++) {
        cout << "Output Channel " << oc << ":\n";
        for (int ic = 0; ic < conv.filters[oc].size(); ic++){
            cout << "\tInput Channel " << ic << ":\n";
            cout << conv.filters[oc][ic] << endl;
        }
    }
    
    cout << "\nInput grads after calc:\n";
    for (int ic = 0; ic < in_grads.size(); ic++) {
        cout << "\tInput Channel " << ic << ":\n";
        cout << in_grads[ic] << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void testing_flatten(){
    Tensor3D input = Tensor3D(3, Tensor2D(2, Tensor1D(4)));
    // Fill with simple increasing pattern
    int val = 1;
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < input[i].size(); j++) {
            for(int k=0 ; k < input[i][j].size() ; k++)
                input[i][j][k] = val++;
        }
    }

    cout << "\nInput:\n";
    for (int ic = 0; ic < input.size(); ic++) {
        cout << "\tInput Channel " << ic << ":\n";
        cout << input[ic] << endl;
    }

    Flatten fl("Flatten layer test");
    Tensor ip = input;
    Tensor1D output = std::get<Tensor1D>(fl.forward(ip));

    cout<<"\nOutput:\n"<<output<<endl;

    cout<<"\nGrads shape\n";
    cout<<'('<<get<0>(fl.input_shape)<<", "<<get<1>(fl.input_shape)<<", "<<get<2>(fl.input_shape)<<" )\n";

    Tensor op = output;
    Tensor3D grads = std::get<Tensor3D>(fl.back_prop(op));
    cout<<"\nReverse:\n";
    for (int ic = 0; ic < grads.size(); ic++) {
        cout << "\tOutput Channel " << ic << ":\n";
        cout << grads[ic] << endl;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void testing_CNN(){
    
    Model test_model = Model({
        new Conv2D("Conv Layer1", {28,28,1}, 3, 4),
        new MaxPool2D("Pool Layer1", {26,26,4}, 2, 2),
        new Conv2D("Conv Layer2", {13,13,4}, 3, 8),
        new MaxPool2D("Pool Layer2", {11,11,8}, 2, 2),
        new Flatten("Flatten layer"),
        new Dense("Dense Layer1", 200, 128),
        new ReLU("ReLU Layer1"),
        new Dense("Dense Layer2", 128, 10),
        new Softmax("softmax1")
    });

    int n_epochs = 5;
    int n_iterations_per_batch = 1;
    int batch_size = 10;
    int dataset_size = 1000;
    double learning_rate = 0.1;
    // Calculate the total number of batches for 1 epoch
    int total_batches = (dataset_size + batch_size - 1) / batch_size;
    string file_path = "./archive/mnist_train.csv";

    cout<<"\n=====Testing with random input before training=====\n";
    for(int i=0 ; i<5 ; i++){
        int x = rand()%dataset_size;
        Batch data = load_mnist_batch(file_path, 1, x);
        data = reshape_mnist_batch(data);
        Tensor1D op = std::get<Tensor1D>(test_model.run(data[0].first));
        cout<<setw(20)<<"Predicted output: [ ";
        for(double d: op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
        Tensor1D actual_op = std::get<Tensor1D>(data[0].second);
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: actual_op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
    }

    indicators::show_console_cursor(false);
    indicators::ProgressBar p{
        indicators::option::BarWidth{20},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::End{"]"},
        indicators::option::ForegroundColor{indicators::Color::yellow},
        indicators::option::PrefixText{"Epoch: 0"},
        indicators::option::PostfixText{"Batch: 0 Loss: inf"},
        indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}},
        indicators::option::MaxProgress{static_cast<size_t>(total_batches * n_epochs)}
    };
    for(int epoch=0 ; epoch<n_epochs ; epoch++){
        p.set_option(indicators::option::PrefixText{"Epoch: "+to_string(epoch)});
        int start_line=1, i=1;
        vector<double> loses;
        while(start_line < dataset_size){
            Batch batch;
            if(start_line + batch_size > dataset_size){
                batch = load_mnist_batch(file_path, dataset_size-start_line, start_line);
            }
            else{
                batch = load_mnist_batch(file_path, batch_size, start_line);
            }
            batch = reshape_mnist_batch(batch);
            vector<double> l = test_model.train(
                batch,                          // batch
                n_iterations_per_batch,         // iterations
                learning_rate                   // learning rate
            );
            loses.insert(loses.end(),l.begin(),l.end());
            // cout<<"loss: "<<l.back()<<"\r";
            p.set_option(indicators::option::PostfixText{"Batch: "+to_string(i)+" Batch Loss: "+to_string(l.back())});
            p.tick();
            i++;
            start_line += batch_size;
        }
        cout<<endl;
    }
    indicators::show_console_cursor(true);
    std::cout << "\033[0m" << std::flush;  // reset colors to default


    cout<<"\n=====Testing with random input after training=====\n";
    for(int i=0 ; i<5 ; i++){
        int x = rand()%dataset_size;
        Batch data = load_mnist_batch(file_path, 1, x);
        data = reshape_mnist_batch(data);
        Tensor1D op = std::get<Tensor1D>(test_model.run(data[0].first));
        cout<<setw(20)<<"Predicted output: [ ";
        for(double d: op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
        Tensor1D actual_op = std::get<Tensor1D>(data[0].second);
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: actual_op)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(){
    // testing ANN
    // test_ann();

    // testing Operations
    // test_op();

    // testing convolutions
    // test_convolution();

    // testing flatten
    // testing_flatten();

    // tesing CNN
    testing_CNN();
}