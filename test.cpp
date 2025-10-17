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
using Batch = vector<pair<vector<double>,vector<double>>>;
using Data = pair<vector<double>,vector<double>>;


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
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: data.second)
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
        cout<<setw(20)<<"Actual output: [ ";
        for(double d: data.second)
            cout<<d<<", ";
        cout<<"\b\b ]\n";
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void test_convolution(){
    
    // Define input dimensions: (height, width, channels)
    tuple<int,int,int> input_dim = {4, 4, 1};

    // Create Conv2d layer
    Conv2d conv("TestConv", input_dim, 3, 1, 1, 0, 1); 
    // filter_size, no_of_filters, stride, padding, dilation
    cout<<get<0>(conv.output_dim)<<','<<get<1>(conv.output_dim)<<','<<get<2>(conv.output_dim)<<'\n';

    // Create simple input (1 channel of 4x4)
    vector<vector<vector<double>>> input(1, vector<vector<double>>(get<0>(input_dim), vector<double>(get<1>(input_dim))));

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
    auto output = std::get<Tensor3D>(conv.forward(input));

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
    
    Tensor3D grads = {{
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

void testing_flatten(){
    Tensor3D ip = Tensor3D(3, Tensor2D(2, Tensor1D(2)));
    // Fill with simple increasing pattern
    int val = 1;
    for (int i = 0; i < ip.size(); i++) {
        for (int j = 0; j < ip[i].size(); j++) {
            for(int k=0 ; k < ip[i][j].size() ; k++)
                ip[i][j][k] = val++;
        }
    }

    cout << "\nInput:\n";
    for (int ic = 0; ic < ip.size(); ic++) {
        cout << "\tInput Channel " << ic << ":\n";
        cout << ip[ic] << endl;
    }

    Flatten fl("Flatten layer test");
    Tensor1D op = std::get<Tensor1D>(fl.forward(ip));

    cout<<"\nOutput:\n"<<op<<endl;

    cout<<"\nGrads shape\n";
    cout<<'('<<get<0>(fl.input_shape)<<", "<<get<1>(fl.input_shape)<<", "<<get<2>(fl.input_shape)<<" )\n";

    Tensor3D grads = std::get<Tensor3D>(fl.back_prop(op));
    cout<<"\nReverse:\n";
    for (int ic = 0; ic < grads.size(); ic++) {
        cout << "\tOutput Channel " << ic << ":\n";
        cout << grads[ic] << endl;
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
}