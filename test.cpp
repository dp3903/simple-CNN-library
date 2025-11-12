#define NOMINMAX
#define NODATA
#include "indicators.hpp"
#include "MLlib.hpp"
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
    size_t height = 28, width = 28, channels = 1;
    Flatten fl;
    fl.input_shape = {channels, height, width}; // we use reverse flatten i.e. backprop of flatten to unflatten the input.
    for(Data d: batch){
        pair<Tensor,Tensor> temp;
        temp.second = d.second;
        temp.first = fl.back_prop(d.first);
        output.push_back(temp);
    }
    return output;
}

void display_mnist_image(Tensor2D image){
    // clear initial colors if any
    cout << "\033[0m\n";

    for(int i=0 ; i < 28 ; i++){
        for(int j=0 ; j < 28 ; j++){
            int x = image[i][j] * 255;
            cout<<"\033[48;2;"<<x<<";"<<x<<";"<<x<<"m ";
            cout<<"\033[48;2;"<<x<<";"<<x<<";"<<x<<"m ";
            cout<<"\033[48;2;"<<x<<";"<<x<<";"<<x<<"m ";
        }
        cout << "\033[0m\n";
    }
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

    SGD optim = SGD(test_model.layers, 0.1);
    
    int n_epochs = 5;
    int n_iterations_per_batch = 1;
    int batch_size = 10;
    int dataset_size = 1000;
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
                /*Batch*/                   batch,
                /*iterations per batch*/    n_iterations_per_batch
            );
            optim.step();
            optim.zero_grad();
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
    cout<<(conv.output_shape.width)<<','<<(conv.output_shape.height)<<','<<(conv.output_shape.channels)<<'\n';

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
    cout<<'('<<(fl.input_shape.width)<<", "<<(fl.input_shape.height)<<", "<<(fl.input_shape.channels)<<" )\n";

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

    Adam optim = Adam(test_model.layers, 0.1);

    cout<<"Input shape: "<<test_model.layers[0]->input_shape<<endl;
    int n_epochs = 5;
    int n_iterations_per_batch = 1;
    int batch_size = 10;
    int dataset_size = 1000;
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
                /*Batch*/                   batch,
                /*iterations per batch*/    n_iterations_per_batch
            );
            optim.step();
            optim.zero_grad();
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

void testing_auto_initializer(){
    Model test_model = Model({
        new Conv2D("Conv Layer1", 3, 4),
        new MaxPool2D("Pool Layer1", 2, 2),
        new Conv2D("Conv Layer2", 3, 8),
        new MaxPool2D("Pool Layer2", 2, 2),
        new Flatten("Flatten layer"),
        new Dense("Dense Layer1", 128),
        new ReLU("ReLU Layer1"),
        new Dense("Dense Layer2", 10),
        new Softmax("softmax1")
    });

    GlobalInitializerStrategies.compile(test_model.layers, {1, 28, 28});

    test_model.set_traianable(false);

    test_model.summary();

    int start_line = 1;
    int batch_size = 10;
    int dataset_size = 1000;
    double learning_rate = 0.1;
    // Calculate the total number of batches for 1 epoch
    int total_batches = (dataset_size + batch_size - 1) / batch_size;
    string file_path = "./archive/mnist_train.csv";
    Batch batch;
    if(start_line + batch_size > dataset_size){
        batch = load_mnist_batch(file_path, dataset_size-start_line, start_line);
    }
    else{
        batch = load_mnist_batch(file_path, batch_size, start_line);
    }
    batch = reshape_mnist_batch(batch);

    // testing forward pass
    vector<double> l = test_model.train(
        /*Batch*/                   batch,
        /*iterations per batch*/    1
    );
}


void testing_model_saving(){
    Model test_model = Model(
        "test_model_1",
        {
            new Conv2D("Conv Layer1", 3, 4),
            new MaxPool2D("Pool Layer1", 2, 2),
            new Conv2D("Conv Layer2", 3, 8),
            new MaxPool2D("Pool Layer2", 2, 2),
            new Flatten("Flatten layer"),
            new Dense("Dense Layer1", 128),
            new ReLU("ReLU Layer1"),
            new Dense("Dense Layer2", 10),
            new Softmax("softmax1")
        }
    );
    GlobalInitializerStrategies.compile(test_model.layers, {1, 28, 28});

    string model_path = "./test_model.bin";

    save_model(test_model, model_path);
    cout<<"Model saved\n";

    Model new_model;
    load_model(new_model, model_path);
    new_model.summary();

    cout<<((Conv2D*)(new_model.layers[0].get()))->filters[0][0][0][0]<<endl;
    cout<<((Conv2D*)(test_model.layers[0].get()))->filters[0][0][0][0]<<endl;
    cout<<((Conv2D*)(new_model.layers[2].get()))->filters[0][0][0][0]<<endl;
    cout<<((Conv2D*)(test_model.layers[2].get()))->filters[0][0][0][0]<<endl;
    cout<<((Dense*)(new_model.layers[5].get()))->weights[0][0]<<endl;
    cout<<((Dense*)(test_model.layers[5].get()))->weights[0][0]<<endl;
    cout<<((Dense*)(new_model.layers[7].get()))->weights[0][0]<<endl;
    cout<<((Dense*)(test_model.layers[7].get()))->weights[0][0]<<endl;
}

void testing_image_display(){
    int n_images = 10;
    Batch b = load_mnist_batch("./archive/mnist_train.csv", n_images, 2);
    for(int i=0 ; i < n_images ; i++){
        Tensor img = b[i].first;
        Flatten fl;
        fl.input_shape = {0, 28, 28};
        Tensor2D reshaped_img = std::get<Tensor2D>(fl.back_prop(img));
        display_mnist_image(reshaped_img);
        cout<<endl;
    }
}

void testing_GAN(){
    Model generator = Model({
        new Dense("generator dense layer 1",20,128), // 10 labels + 10 noise
        new ReLU("generator relu layer 1"),
        new Dense("generator dense layer 2",128,256),
        new ReLU("generator relu layer 2"),
        new Dense("generator dense layer 3",256,784),
        new Tanh("generator tanh layer")
    });

    Model descriminator = Model({
        new Dense("descriminator dense layer 1", 794, 256), // 784 pixels + 10 labels for cGAN
        new ReLU("descriminator relu layer 1"),
        new Dense("descriminator dense layer 2", 256, 128),
        new ReLU("descriminator relu layer 2"),
        new Dense("descriminator dense layer 3", 128, 32),
        new ReLU("descriminator relu layer 3"),
        new Dense("descriminator dense layer 4", 32, 1),
        new Sigmoid("descriminator sigmoid layer")
    });

    Adam gen_optim = Adam(generator.layers, 0.0002, 0.5);
    Adam desc_optim = Adam(descriminator.layers, 0.0002, 0.5);
    
    int n_epochs = 100;
    int n_desc_iterations_per_batch = 1;
    int batch_size = 100;
    int dataset_size = 1000;
    int noise_vector_size = 10;
    // Calculate the total number of batches for 1 epoch
    int total_batches = (dataset_size + batch_size - 1) / batch_size;
    string file_path = "./archive/mnist_train.csv";

    for(int epoch=0 ; epoch<n_epochs ; epoch++){
        int start_line=1, i=1;
        vector<double> loses;
        while(start_line < dataset_size){
            Batch desc_batch;
            Batch gen_batch;
            Batch batch;

            if(start_line + batch_size > dataset_size){
                batch = load_mnist_batch(file_path, dataset_size-start_line, start_line);
            }
            else{
                batch = load_mnist_batch(file_path, batch_size, start_line);
            }

            for(auto& d: batch){
                Tensor noise = Tensor1D(noise_vector_size);
                Tensor1D& t_noise = std::get<Tensor1D>(noise);
                Tensor1D& pixels = std::get<Tensor1D>(d.first);
                Tensor1D& labels = std::get<Tensor1D>(d.second);
                pixels = pixels * 2; // (0,1) -> (0,2)
                pixels = pixels - 1; // (0,2) -> (-1,1)
                for(auto& n: t_noise)
                    n = generate_random_in_range(-50, 50);
               
                pixels.insert(pixels.end(), labels.begin(), labels.end()); // append labels to pixels
                t_noise.insert(t_noise.end(), labels.begin(), labels.end()); // append labels to noise
                desc_batch.push_back(Data(pixels, Tensor1D(1, 1)));
                gen_batch.push_back(Data(noise, Tensor1D(1, 1)));
                Tensor gen_op = generator.run(noise);
                Tensor1D& t = std::get<Tensor1D>(gen_op);
                t.insert(t.end(), labels.begin(), labels.end());
                desc_batch.push_back(Data(gen_op, Tensor1D(1, 0)));
            }

            descriminator.set_traianable(true);
            vector<double> l = descriminator.train(
                /*Batch*/                   desc_batch,
                /*iterations per batch*/    n_desc_iterations_per_batch
            );
            desc_optim.step();
            desc_optim.zero_grad();
            
            descriminator.set_traianable(false);
            for(Data& d: gen_batch){
                Tensor1D n = std::get<Tensor1D>(d.first);
                Tensor gen_op = generator.run(d.first);
                Tensor1D& t = std::get<Tensor1D>(gen_op);
                t.insert(t.end(), n.begin()+noise_vector_size, n.end()); // append label to pixels
                Tensor op = descriminator.run(gen_op);
                BinaryCrossEntropyLoss l;
                auto [loss, grads] = l.calculate(op, d.second);
                Tensor desc_grads = descriminator.back(grads);
                std::get<Tensor1D>(desc_grads).resize(784);
                generator.back(desc_grads);
            }
            gen_optim.step();
            gen_optim.zero_grad();

            cout<<"\rBatch: "<<i<<'/'<<total_batches;
            i++;
            start_line += batch_size;
        }
        cout<<"\nEpoch: "<<epoch<<endl;
        for(int u=0 ; u<10 ; u++){
            Tensor1D t_demo = Tensor1D(10,0);
            t_demo[u] = 1;
            Tensor1D t_noise = Tensor1D(noise_vector_size);
            for(auto& n: t_noise)
                n = generate_random_in_range(-50, 50);
            t_noise.insert(t_noise.end(), t_demo.begin(), t_demo.end());
            Tensor demo = t_noise;
            Tensor op = generator.run(demo);
            op = op + 1; // (-1,1) -> (0,2)
            op = op / 2; // (0,2) -> (0,1)
            Flatten fl;
            fl.input_shape = {0, 28, 28};
            display_mnist_image(std::get<Tensor2D>(fl.back_prop(op)));
        }
    }
}

void testing_custom_model(){
    Model test_model = Model({
        new Dense("Layer1", 784, 256),
        new ReLU("relu1"),
        new Dense("Layer2", 256, 128),
        new ReLU("relu2"),
        new Dense("Layer3", 128, 10),
        new Softmax("softmax1")
    });

    SGD optim = SGD(test_model.layers, 0.005);
    
    int n_epochs = 5;
    int n_iterations_per_batch = 1;
    int batch_size = 10;
    int dataset_size = 1000;
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
                /*Batch*/                   batch,
                /*iterations per batch*/    n_iterations_per_batch
            );
            optim.step();
            optim.zero_grad();
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

    cout<<"\n=====Generating the image=====\n";
    int n_image_gen_iterations = 20;
    double ip_learning_rate = 0.1;
    test_model.set_traianable(false);
    Tensor expected = Tensor1D({0,1,0,0,0,0,0,0,0,0});
    Tensor1D noise = Tensor1D(784);
    Flatten fl;
    fl.input_shape = {0, 28, 28};
    for(auto& d: noise)
        d = generate_random_in_range(0,1);
    Tensor t_noise = noise;
    for(int i=0 ; i < n_image_gen_iterations ; i++){
        cout<<"Iteration "<<i+1<<'/'<<n_image_gen_iterations<<endl;
        Tensor t_op = test_model.run(t_noise);
        auto [loss, grads] = test_model.loss->calculate(t_op, expected);
        Tensor ip_grads = test_model.back(grads);
        t_noise = t_noise - (ip_grads * ip_learning_rate);
        display_mnist_image(get<Tensor2D>(fl.back_prop(t_noise)));
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
    // testing_CNN();

    // testing auto initializer
    // testing_auto_initializer();
    
    // testing model save and load 
    // testing_model_saving();

    // testing image display in terminal
    // testing_image_display();

    // testing simple linear GAN
    // testing_GAN();

    // testing custom model
    testing_custom_model();
}