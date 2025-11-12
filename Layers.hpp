#pragma once
#include <vector>
#include <iostream>
#include <variant>
#include <iomanip>
#include "Overloads.hpp"


double generate_random_in_range(double low, double high){
    if(high < low)
        swap(high,low);
    return (double(rand()) / RAND_MAX)*(high-low) + low;
}


class Layer{
    public:
        string label;
        bool trainable = true;
        Shape input_shape;
        Shape output_shape;

        Layer() = default;

        Layer(string name){
            this->label = name;
        }

        virtual Tensor forward(Tensor input){
            throw runtime_error("Invalid input for layer: "+this->label+".");
        }

        virtual Tensor back_prop(Tensor grads){
            throw runtime_error("Invalid gradients for layer: "+this->label+".");
        }
        
        virtual void update_weights(const std::function<void(Value&, int&)>& visitor) = 0;
        virtual void compute_output_shape() = 0;
        virtual int parameter_count() = 0;

        // You need a serialize function in the base class
        template <class Archive>
        void serialize(Archive& archive) {
            archive(
                CEREAL_NVP(label),   
                CEREAL_NVP(input_shape),
                CEREAL_NVP(output_shape),
                CEREAL_NVP(trainable)
            ); // Example: serialize base members
        }
};

class Dense : public Layer{
    public:
        vector<vector<Value>> weights;
        vector<Value> biases;
        Tensor1D input;

        Dense() = default;

        Dense(string label, size_t output_size) : Layer(label){
            this->weights = vector<vector<Value>>(output_size);
            this->biases = vector<Value>(output_size);
            this->output_shape = {0, 0, output_size};
        }

        Dense(string label, size_t input_size, size_t output_size) : Dense(label, output_size){
            this->input = vector<double>(input_size);
            this->input_shape = {0, 0, input_size};
            compute_output_shape();
        }

        void compute_output_shape() override{
            if(weights.empty())
                throw invalid_argument("Cannot compute output shape before initialization.");
            
            for(int i=0 ; i < output_shape.width ; i++){
                weights[i] = vector<Value>(input_shape.width);
                biases[i] = Value("Dense: "+this->label+" bias("+to_string(i)+").", generate_random_in_range(-0.1, 0.1));
                for(int j=0 ; j < input_shape.width ; j++){
                    weights[i][j] = Value("Dense: "+this->label+" param("+to_string(i)+','+to_string(j)+").", generate_random_in_range(-0.1, 0.1));
                }
            }
        }

        int parameter_count() override{
            if(weights.empty() || weights[0].empty() || biases.empty())
                return -1;

            return (input_shape.width * output_shape.width + output_shape.width);
        }

        Tensor forward(Tensor t_input) override {
            if(!holds_alternative<Tensor1D>(t_input))
                throw runtime_error("Invalid input Tensor to Dense layer: "+this->label+".");
            
            const Tensor1D& input = std::get<Tensor1D>(t_input);
            if(!weights.size() || !input.size()){
                throw runtime_error("Cannot forword on empty layer for layer: "+this->label+".");
            }
            if(input_shape.width != input.size())
                throw runtime_error("Input size does not match expected input size for layer: "+this->label+". expected "+to_string(input_shape.width)+" recieved "+to_string(input.size())+".");
                
            this->input = input;
            vector<double> op = vector<double>(weights.size(), 0);
                    
            try{
                for(int i=0 ; i < weights.size() ; i++){
                    op[i] = biases[i].val;
                    for(int j=0 ; j < weights[i].size() ; j++){
                        op[i] += (input[j] * weights[i][j].val);
                    }
                }

                // cout<<this->label<<":[ ";
                // for(double d: op)
                //     cout<<d<<", ";
                // cout<<"\b\b ]\n";

                return op;
            }
            catch(string s){
                throw runtime_error("Error in forwarding from dense layer: '"+this->label+"'.");
            }
        }

        Tensor back_prop(Tensor t_grads) override{
            if(!holds_alternative<Tensor1D>(t_grads))
                throw runtime_error("Invalid gradient Tensor to Dense layer: "+this->label+".");
            
            const Tensor1D& grads = std::get<Tensor1D>(t_grads);
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads for layer: "+this->label+".");
            if(grads.size() != weights.size())
                throw runtime_error("Input grads size does not match no. of neurons for layer: "+this->label+".");

            vector<double> op = vector<double>(weights[0].size(), 0);
            for(int i=0 ; i<weights.size() ; i++){
                (this->trainable) && (biases[i].grad += grads[i]);
                for(int j=0 ; j<weights[i].size() ; j++){
                    (this->trainable) && (weights[i][j].grad += input[j] * grads[i]);
                    op[j] += weights[i][j].val * grads[i];
                }
            }

            // cout<<this->label<<":[ ";
            // for(double d: op)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

            return op;
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override{
            int index=0;
            for (auto& row : weights) {
                for (auto& w : row) {
                    // Apply the visitor function to each weight
                    visitor(w,index);
                    index++;
                }
            }
            for (auto& b : biases) {
                // Apply the visitor function to each bias
                visitor(b,index);
                index++;
            }
        }

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
            archive(
                CEREAL_NVP(weights),    // Automatically handles the vector of vectors
                CEREAL_NVP(biases)      // Automatically handles the vector
            );
        }
};

class Conv2D : public Layer{
    public:
        vector<vector<vector<vector<Value>>>> filters;
        vector<Value> biases;
        Tensor3D input;
        int stride;
        int padding;
        int filter_size;
        int dilation;

        Conv2D() = default;

        Conv2D(
            string label,
            int filter_size,
            int no_of_filters,
            int stride=1,
            int padding=0,
            int dilation=1
        ){
            if(stride <= 0)
                throw invalid_argument("Stride cannot be 0 or negative for Conv2D layer: "+label);
            if(padding < 0)
                throw invalid_argument("Padding cannot be negative for Conv2D layer: "+label);

            this->label = label;
            this->stride = stride;
            this->filter_size = filter_size;
            this->padding = padding;
            this->dilation = dilation;
            this->filters = vector<vector<vector<vector<Value>>>>(no_of_filters);
            this->biases = vector<Value>(no_of_filters);
        }

        Conv2D(
            string label,
            tuple<size_t, size_t, size_t> input_dim,
            int filter_size,
            int no_of_filters,
            int stride=1,
            int padding=0,
            int dilation=1
        ) : Conv2D(label, filter_size, no_of_filters, stride, padding, dilation){
            if(get<0>(input_dim)<=0 || get<1>(input_dim)<=0 || get<2>(input_dim)<=0)
                throw invalid_argument("Input dimension cannot be less than 0 for Conv2D layer: "+label);
            this->input_shape = {
                get<2>(input_dim),
                get<1>(input_dim),
                get<0>(input_dim)
            };
            compute_output_shape();
            
        }

        void compute_output_shape() override{
            int effective_filter_size = dilation * (filter_size - 1) + 1;
            int no_of_filters = filters.size();
            this->output_shape = {
                (size_t)no_of_filters,
                ((input_shape.height + 2*padding - effective_filter_size) / stride + 1),
                ((input_shape.width + 2*padding - effective_filter_size) / stride + 1)
            };
            this->filters = vector<vector<vector<vector<Value>>>>(no_of_filters, vector<vector<vector<Value>>>(input_shape.channels, vector<vector<Value>>(filter_size, vector<Value>(filter_size))));
            this->input = vector<vector<vector<double>>>(input_shape.channels, vector<vector<double>>(input_shape.height, vector<double>(input_shape.width)));
        
            for(int i=0 ; i < filters.size() ; i++){
                biases[i] = Value(
                    "Conv2D: "+this->label+" bias("+to_string(i)+").",
                    generate_random_in_range(-0.1, 0.1)
                );
                for(int j=0 ; j < filters[i].size() ; j++){
                    for(int k=0 ; k < filters[i][j].size() ; k++){
                        for(int l=0 ; l < filters[i][j][k].size() ; l++){
                            filters[i][j][k][l] = Value(
                                "Conv2D: "+this->label+" param("+to_string(i)+','+to_string(j)+','+to_string(k)+','+to_string(l)+").",
                                generate_random_in_range(-0.1, 0.1)
                            );
                        }
                    }
                }
            }
        }

        int parameter_count() override{
            if(filters.empty() || filters[0].empty() || filters[0][0].empty() || filters[0][0][0 ].empty() || biases.empty())
                return -1;
            return (filters.size() * filters[0].size() * filters[0][0].size() * filters[0][0][0].size() + biases.size());
        }

        Tensor forward(Tensor t_input) override {
            if(!holds_alternative<Tensor3D>(t_input))
                throw runtime_error("Invalid input Tensor to Conv2D layer: "+this->label+".");
            
            const Tensor3D& input = std::get<Tensor3D>(t_input);
            if(input.size() != input_shape.channels)
                throw runtime_error("Invalid input channel dimensions for Conv2D layer: "+this->label+". expected '"+to_string(input_shape.channels)+"' recieved '"+to_string(input.size())+"'.");
            
            Tensor3D ans = Tensor3D(output_shape.channels, Tensor2D(output_shape.width, Tensor1D(output_shape.height,0)));
            for (int oc = 0; oc < output_shape.channels; oc++) {
                vector<vector<double>> ans_oc(output_shape.width, vector<double>(output_shape.height, 0));
                for (int ic = 0; ic < input_shape.channels; ic++) {
                    ans_oc = ans_oc + convolve(input[ic], filters[oc][ic], padding, stride, dilation);
                }
                ans[oc] = (ans_oc + biases[oc].val);
            }

            this -> input = input;

            return ans;
        }
        
        Tensor back_prop(Tensor t_grads) override{
            if(!holds_alternative<Tensor3D>(t_grads))
                throw runtime_error("Invalid gradient Tensor to Conv2D layer: "+this->label+".");
            
            const Tensor3D& grads = std::get<Tensor3D>(t_grads);
            if(grads.size() != output_shape.channels)
                throw runtime_error("Invalid no. of channels in output grads for layer: "+this->label);

            vector<vector<vector<double>>> ans(input_shape.channels, vector<vector<double>>(input_shape.height, vector<double>(input_shape.width, 0)));
            for (int oc = 0; oc < output_shape.channels; oc++) {
                if(grads[oc].size() != output_shape.width)
                    throw runtime_error("Invalid x-dim in output grads for layer: "+this->label);
                for (int out_x = 0; out_x < output_shape.width; out_x++) {
                    if(grads[oc][out_x].size() != output_shape.height)
                        throw runtime_error("Invalid y-dim in output grads for layer: "+this->label);
                    for (int out_y = 0; out_y < output_shape.height; out_y++) {
                        (this->trainable) && (biases[oc].grad += grads[oc][out_x][out_y]);
                        for (int ic = 0; ic < input_shape.channels; ic++) {
                            for (int i = 0; i < filter_size; i++) {
                                for (int j = 0; j < filter_size; j++) {
                                    int in_x = out_x * stride + dilation * i - padding;
                                    int in_y = out_y * stride + dilation * j - padding;
                                    if (in_x >= 0 && in_x < input_shape.width && in_y >= 0 && in_y < input_shape.height) {
                                        ans[ic][in_x][in_y] += filters[oc][ic][i][j].val * grads[oc][out_x][out_y];
                                        (this->trainable) && (filters[oc][ic][i][j].grad += input[ic][in_x][in_y] * grads[oc][out_x][out_y]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return ans;
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override{
            int index=0;
            for(auto& oc: filters){
                for(auto& ic: oc){
                    for(auto& row: ic){
                        for(auto& w: row){
                            visitor(w,index);
                            index++;
                        }
                    }
                }
            }

            for(auto& b: biases){
                visitor(b,index);
                index++;
            }
        }

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
            archive(
                CEREAL_NVP(filters),    // Automatically handles the vector of vectors
                CEREAL_NVP(biases)      // Automatically handles the vector
            );
        }
};

class Flatten : public Layer{
    public:
        Flatten(){}

        Flatten(string label){
            this->label = label;
        }

        void compute_output_shape() override{
            output_shape = {0, 0, input_shape.total_elements()};
        }

        int parameter_count() override{
            return 0;
        }

        Tensor forward(Tensor t_input) override  {
            Tensor1D output;
            if(holds_alternative<Tensor1D>(t_input)){
                const Tensor1D& input = std::get<Tensor1D>(t_input);
                this->input_shape = {0, 0, input.size()};
                output = input;
            }
            else if(holds_alternative<Tensor2D>(t_input)){
                const Tensor2D& input = std::get<Tensor2D>(t_input);
                this->input_shape = {0, input.size(), input[0].size()};
                for(const Tensor1D& row: input){
                    output.insert(output.end(), row.begin(), row.end());
                }
            }
            else if(holds_alternative<Tensor3D>(t_input)){
                const Tensor3D& input = std::get<Tensor3D>(t_input);
                this->input_shape = {input.size(), input[0].size(), input[0][0].size()};
                for(const Tensor2D& channel: input){
                    for(const Tensor1D& row: channel){
                        output.insert(output.end(), row.begin(), row.end());
                    }
                }
            }
            return output;
        }

        Tensor back_prop(Tensor t_grads) override {
            if(!holds_alternative<Tensor1D>(t_grads))
                throw runtime_error("Invalid gradient Tensor to Flatten layer: "+this->label+".");

            const Tensor1D& grads = std::get<Tensor1D>(t_grads);
            if(input_shape.total_elements() != grads.size())
                throw runtime_error("Invalid size of gradient Tensor1D to Flatten layer: "+this->label+".");

            // int x=get<0>(input_shape);
            // int y=get<1>(input_shape);
            // int z=get<2>(input_shape);
            if(input_shape.channels == 0 && input_shape.height == 0){
                return grads;
            }
            else if(input_shape.channels == 0){
                Tensor2D ip_grads = Tensor2D(input_shape.height, Tensor1D(input_shape.width));
                for(int i=0 ; i<input_shape.height ; i++)
                    for(int j=0 ; j<input_shape.width ; j++)
                        ip_grads[i][j] = grads[i*input_shape.width + j];
                return ip_grads;
            }
            else{
                Tensor3D ip_grads = Tensor3D(input_shape.channels, Tensor2D(input_shape.height, Tensor1D(input_shape.width)));
                for(int i=0 ; i<input_shape.channels ; i++)
                    for(int j=0 ; j<input_shape.height ; j++)
                        for(int k=0 ; k<input_shape.width ; k++)
                            ip_grads[i][j][k] = grads[i*input_shape.height*input_shape.width + j*input_shape.width + k];
                return ip_grads;
            }
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override{}

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
        }
};

class MaxPool2D : public Layer {
    public:
        // This will store the (x, y) coordinates of the max value for each output pixel
        vector<vector<vector<pair<int, int>>>> max_indices;
        
        int stride;
        int padding;
        int filter_size;
        int dilation;

        MaxPool2D() = default;

        MaxPool2D(
            string label,
            int filter_size,
            int stride = 1,
            int padding = 0,
            int dilation = 1
        ) {
            this->label = label;
            this->filter_size = filter_size;
            this->stride = stride;
            this->padding = padding;
            this->dilation = dilation;
        }

        MaxPool2D(string label,
            tuple<size_t, size_t, size_t> input_dim,
            int filter_size,
            int stride = 1,
            int padding = 0,
            int dilation = 1
        ) : MaxPool2D(label, filter_size, stride, padding, dilation) {
            // ... (Your constructor is correct, no changes needed here) ...
            this->input_shape = {
                get<2>(input_dim),
                get<1>(input_dim),
                get<0>(input_dim)
            };
            compute_output_shape();
        }

        void compute_output_shape() override{
            int effective_filter_size = dilation * (filter_size - 1) + 1;
            this->output_shape = {
                input_shape.channels,
                ((input_shape.height + 2*padding - effective_filter_size) / stride + 1),
                ((input_shape.width + 2*padding - effective_filter_size) / stride + 1)
            };
        }

        int parameter_count() override{
            return 0;
        }

        // FIX: Pass input by const reference to avoid expensive copy
        Tensor forward(Tensor t_input) override  {
            if (!holds_alternative<Tensor3D>(t_input))
                throw invalid_argument("Invalid input Tensor to MaxPool2D layer: " + this->label);

            // FIX: Get a const reference, don't make a full copy for 'this->input'
            const Tensor3D& input = std::get<Tensor3D>(t_input);

            auto [channels, out_h, out_w] = this->output_shape;
            Tensor3D output(channels, Tensor2D(out_h, Tensor1D(out_w)));
            
            // Initialize the mask to store indices
            this->max_indices.assign(channels, vector<vector<pair<int, int>>>(out_h, vector<pair<int, int>>(out_w)));

            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        double max_val = -numeric_limits<double>::infinity();
                        int max_x = -1, max_y = -1;

                        for (int i = 0; i < filter_size; ++i) {
                            for (int j = 0; j < filter_size; ++j) {
                                int in_x = h * stride + i * dilation - padding;
                                int in_y = w * stride + j * dilation - padding;

                                if (in_x >= 0 && in_x < input_shape.height && in_y >= 0 && in_y < input_shape.width) {
                                    if (input[c][in_x][in_y] > max_val) {
                                        max_val = input[c][in_x][in_y];
                                        max_x = in_x;
                                        max_y = in_y;
                                    }
                                }
                            }
                        }
                        output[c][h][w] = max_val;
                        // FIX: Store the index of the max value
                        this->max_indices[c][h][w] = {max_x, max_y};
                    }
                }
            }
            return output;
        }

        // FIX: Pass grads by const reference
        Tensor back_prop(Tensor t_grads) override {
            if (!holds_alternative<Tensor3D>(t_grads))
                throw invalid_argument("Invalid grads Tensor to MaxPool2D layer: " + this->label);

            const Tensor3D& grads = std::get<Tensor3D>(t_grads);
            
            auto [channels, in_h, in_w] = this->input_shape;
            auto [_,out_h, out_w] = this->output_shape;

            // Initialize input gradients to all zeros
            Tensor3D input_grads(channels, Tensor2D(in_h, Tensor1D(in_w, 0.0)));

            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < out_h; ++h) {
                    for (int w = 0; w < out_w; ++w) {
                        // FIX: Directly look up the saved index
                        int max_x = this->max_indices[c][h][w].first;
                        int max_y = this->max_indices[c][h][w].second;

                        if (max_x != -1) { // Check if the max was found in a valid region
                            // FIX: Accumulate gradients, don't overwrite
                            input_grads[c][max_x][max_y] += grads[c][h][w];
                        }
                    }
                }
            }
            return input_grads;
        }
        
        void update_weights(const std::function<void(Value&, int&)>& visitor) override {}

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
        }
};

class Softmax : public Layer{
    public: 
        Tensor1D input;

        Softmax(){}

        Softmax(string label){
            this->label = label;
        }

        void compute_output_shape() override{
            output_shape = input_shape;
        }

        int parameter_count() override{
            return 0;
        }

        Tensor forward(Tensor t_input) override {
            if(!holds_alternative<Tensor1D>(t_input))
                throw invalid_argument("Invalid input Tensor to Softmax layer: "+this->label+".");

            this->input = std::get<Tensor1D>(t_input);
            double sum = 0.0;
            Tensor1D op = Tensor1D(input.size());
            for(int i=0 ; i < input.size() ; i++){
                op[i] = exp(input[i]);
                sum += op[i];
            }
            for(int i=0 ; i < input.size() ; i++){
                op[i] = op[i] / sum;
            }

            // cout<<this->label<<" output:[ ";
            // for(double d: op)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

            return op;
        }

        Tensor back_prop(Tensor t_grads) override{
            if(!holds_alternative<Tensor1D>(t_grads))
                throw runtime_error("Invalid input Tensor to Softmax layer: "+this->label+".");
            
            const Tensor1D& grads = std::get<Tensor1D>(t_grads);
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads for layer: "+this->label+".");
            if(grads.size() != input.size())
                throw runtime_error("grads size does not match no. of inputs for layer: "+this->label+".");

            // vector<double> op = vector<double>(grads.size(), 0);
            // double exps[input.size()];
            // double sum=0;
            // for(int i=0 ; i<input.size() ; i++){
            //     exps[i] = exp(input[i]);
            //     sum += exps[i];
            // }
            // for(int i=0 ; i<input.size() ; i++){
            //     for(int j=0 ; j<input.size() ; j++){
            //         op[i] += grads[i] * (exps[i]/sum) * ((i==j ? 1 : 0)-exps[j]/sum);
            //     }
            // }

            // // cout<<this->label<<" grads:[ ";
            // // for(double d: op)
            // //     cout<<d<<", ";
            // // cout<<"\b\b ]\n";

            // return op;

            return grads;
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override {}

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
        }
};

class ReLU : public Layer {
    public:

        // The cache for the input from the forward pass
        Tensor input_cache;

        ReLU() {}
        ReLU(std::string label) {
            this->label = label;
        }

        void compute_output_shape() override{
            output_shape = input_shape;
        }

        int parameter_count() override{
            return 0;
        }

        Tensor forward(Tensor t_input) override  {
            this->input_cache = t_input; // Store original input
            if(holds_alternative<Tensor1D>(t_input)){
                Tensor1D& temp = std::get<Tensor1D>(t_input);
                for(auto& val: temp)
                    if(val < 0)
                        val = 0;
            }
            else if(holds_alternative<Tensor2D>(t_input)){
                Tensor2D& temp = std::get<Tensor2D>(t_input);
                for(auto& row: temp)
                    for(auto& val: row)
                        if(val < 0)
                            val = 0;
            }
            else{
                Tensor3D& temp = std::get<Tensor3D>(t_input);
                for(auto& channel: temp)
                    for(auto& row: channel)
                        for(auto& val: row)
                            if(val < 0)
                                val = 0;
            }
            return t_input;
        }

        Tensor back_prop(Tensor t_grads) override {
            if(holds_alternative<Tensor1D>(t_grads)){
                Tensor1D& temp = std::get<Tensor1D>(input_cache);
                Tensor1D& temp_grads = std::get<Tensor1D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++)
                    if(temp[i] <= 0)
                        temp_grads[i] = 0;
            }
            else if(holds_alternative<Tensor2D>(t_grads)){
                Tensor2D& temp = std::get<Tensor2D>(input_cache);
                Tensor2D& temp_grads = std::get<Tensor2D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++)
                    for(int j=0 ; j < temp[i].size() ; j++)
                        if(temp[i][j] <= 0)
                            temp_grads[i][j] = 0;
            }
            else{
                Tensor3D& temp = std::get<Tensor3D>(input_cache);
                Tensor3D& temp_grads = std::get<Tensor3D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++)
                    for(int j=0 ; j < temp[i].size() ; j++)
                        for(int k=0 ; k < temp[i][j].size() ; k++)
                            if(temp[i][j][k] <= 0)
                                temp_grads[i][j][k] = 0;
            }
            return t_grads;
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override {}

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
        }
};

class Sigmoid : public Layer{
    public:
        Tensor input_cache;

        Sigmoid() {}
        Sigmoid(std::string label) {
            this->label = label;
        }

        void compute_output_shape() override{
            output_shape = input_shape;
        }

        int parameter_count() override{
            return 0;
        }

        Tensor forward(Tensor t_input) override  {
            this->input_cache = t_input; // Store original input
            if(holds_alternative<Tensor1D>(t_input)){
                Tensor1D& temp = std::get<Tensor1D>(t_input);
                for(auto& val: temp)
                    val = 1 / (1 + exp(-val));
            }
            else if(holds_alternative<Tensor2D>(t_input)){
                Tensor2D& temp = std::get<Tensor2D>(t_input);
                for(auto& row: temp)
                    for(auto& val: row)
                        val = 1 / (1 + exp(-val));
            }
            else{
                Tensor3D& temp = std::get<Tensor3D>(t_input);
                for(auto& channel: temp)
                    for(auto& row: channel)
                        for(auto& val: row)
                            val = 1 / (1 + exp(-val));
            }
            return t_input;
        }

        Tensor back_prop(Tensor t_grads) override {
            if(holds_alternative<Tensor1D>(t_grads)){
                Tensor1D& temp = std::get<Tensor1D>(input_cache);
                Tensor1D& temp_grads = std::get<Tensor1D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++){
                    double x = exp(-temp[i]);
                    temp_grads[i] *= x / pow(1 + x, 2);
                }
            }
            else if(holds_alternative<Tensor2D>(t_grads)){
                Tensor2D& temp = std::get<Tensor2D>(input_cache);
                Tensor2D& temp_grads = std::get<Tensor2D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++){
                    for(int j=0 ; j < temp[i].size() ; j++){
                        double x = exp(-temp[i][j]);
                        temp_grads[i][j] *= x / pow(1 + x, 2);
                    }
                }
            }
            else{
                Tensor3D& temp = std::get<Tensor3D>(input_cache);
                Tensor3D& temp_grads = std::get<Tensor3D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++){
                    for(int j=0 ; j < temp[i].size() ; j++){
                        for(int k=0 ; k < temp[i][j].size() ; k++){
                            double x = exp(-temp[i][j][k]);
                            temp_grads[i][j][k] *= x / pow(1 + x, 2);
                        }
                    }
                }
            }
            return t_grads;
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override {}

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
        }
};

class Tanh : public Layer{
    public:
        Tensor input_cache;

        Tanh() {}
        Tanh(std::string label) {
            this->label = label;
        }

        void compute_output_shape() override{
            output_shape = input_shape;
        }

        int parameter_count() override{
            return 0;
        }

        Tensor forward(Tensor t_input) override  {
            this->input_cache = t_input; // Store original input
            if(holds_alternative<Tensor1D>(t_input)){
                Tensor1D& temp = std::get<Tensor1D>(t_input);
                for(auto& val: temp){
                    val = exp(2*val);
                    val = (val - 1) / (val + 1);
                }
            }
            else if(holds_alternative<Tensor2D>(t_input)){
                Tensor2D& temp = std::get<Tensor2D>(t_input);
                for(auto& row: temp){
                    for(auto& val: row){
                        val = exp(2*val);
                        val = (val - 1) / (val + 1);
                    }
                }
            }
            else{
                Tensor3D& temp = std::get<Tensor3D>(t_input);
                for(auto& channel: temp){
                    for(auto& row: channel){
                        for(auto& val: row){
                            val = exp(2*val);
                            val = (val - 1) / (val + 1);
                        }
                    }
                }
            }
            return t_input;
        }

        Tensor back_prop(Tensor t_grads) override {
            if(holds_alternative<Tensor1D>(t_grads)){
                Tensor1D& temp = std::get<Tensor1D>(input_cache);
                Tensor1D& temp_grads = std::get<Tensor1D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++){
                    double x = exp(2*temp[i]);
                    temp_grads[i] *= 4*x / pow(x*x + 2*x + 1, 2);
                }
            }
            else if(holds_alternative<Tensor2D>(t_grads)){
                Tensor2D& temp = std::get<Tensor2D>(input_cache);
                Tensor2D& temp_grads = std::get<Tensor2D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++){
                    for(int j=0 ; j < temp[i].size() ; j++){
                        double x = exp(2*temp[i][j]);
                        temp_grads[i][j] *= 4*x / pow(x*x + 2*x + 1, 2);
                    }
                }
            }
            else{
                Tensor3D& temp = std::get<Tensor3D>(input_cache);
                Tensor3D& temp_grads = std::get<Tensor3D>(t_grads);
                for(int i=0 ; i < temp.size() ; i++){
                    for(int j=0 ; j < temp[i].size() ; j++){
                        for(int k=0 ; k < temp[i][j].size() ; k++){
                            double x = exp(2*temp[i][j][k]);
                            temp_grads[i][j][k] *= 4*x / pow(x*x + 2*x + 1, 2);
                        }
                    }
                }
            }
            return t_grads;
        }

        void update_weights(const std::function<void(Value&, int&)>& visitor) override {}

        template <class Archive>
        void serialize(Archive& archive) {
            // This is crucial: it calls the base class's serialize function
            archive(cereal::base_class<Layer>(this)); 
        }
};


// Serialization registration for layers for cereal library
CEREAL_REGISTER_TYPE(Dense);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, Dense);
CEREAL_REGISTER_TYPE(Conv2D);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, Conv2D);
CEREAL_REGISTER_TYPE(MaxPool2D);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, MaxPool2D);
CEREAL_REGISTER_TYPE(ReLU);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, ReLU);
CEREAL_REGISTER_TYPE(Softmax);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, Softmax);
CEREAL_REGISTER_TYPE(Flatten);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, Flatten);
CEREAL_REGISTER_TYPE(Sigmoid);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, Sigmoid);
CEREAL_REGISTER_TYPE(Tanh);
CEREAL_REGISTER_POLYMORPHIC_RELATION(Layer, Tanh);