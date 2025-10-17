#include <vector>
#include <iostream>
#include <variant>
#include <iomanip>
#include <type_traits>
#include "Overloads.hpp"


double generate_random_in_range(double low, double high){
    if(high < low)
        swap(high,low);
    return (double(rand()) / RAND_MAX)*(high-low) + low;
}


class Layer{
    public:
        string label;

        Layer(){}
        Layer(string name){
            this->label = name;
        }

        virtual Tensor forward(Tensor input){
            throw runtime_error("Invalid input for layer: "+this->label+".");
        }

        virtual Tensor back_prop(Tensor grads){
            throw runtime_error("Invalid gradients for layer: "+this->label+".");
        }

        
        virtual void update_weights(double learning_rate = 0.1) = 0;
};

class Dense : public Layer{
    public:
        vector<vector<Value>> weights;
        Tensor1D input;

        Dense(){}

        Dense(string label, int input_size, int output_size) : Layer(label){
            this->label = (label.size()!=0 ? label : "temp");
            this->weights = vector<vector<Value>>(output_size, vector<Value>(input_size));
            this->input = vector<double>(input_size);
            for(int i=0 ; i < weights.size() ; i++){
                for(int j=0 ; j < weights[i].size() ; j++){
                    weights[i][j] = Value("Dense: "+this->label+" param("+to_string(i)+','+to_string(j)+").", generate_random_in_range(-0.1, 0.1));
                }
            }
        }

        Tensor forward(const Tensor t_input){
            if(!holds_alternative<Tensor1D>(t_input))
                throw runtime_error("Invalid input Tensor to Dense layer: "+this->label+".");
            
            const Tensor1D& input = std::get<Tensor1D>(t_input);
            if(!weights.size() || !input.size() || !(this->input.size())){
                throw runtime_error("Cannot forword on empty layer for layer: "+this->label+".");
            }
            if(this->input.size() != input.size())
                throw runtime_error("Input size does not match expected input size for layer: "+this->label+".");
                
            this->input = input;
            vector<double> op = vector<double>(weights.size(), 0);
                    
            try{
                for(int i=0 ; i < weights.size() ; i++){
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

        Tensor back_prop(const Tensor t_grads){
            if(!holds_alternative<Tensor1D>(t_grads))
                throw runtime_error("Invalid gradient Tensor to Dense layer: "+this->label+".");
            
            const Tensor1D& grads = std::get<Tensor1D>(t_grads);
            if(!grads.size())
                throw runtime_error("Cannot backprop on empty grads for layer: "+this->label+".");
            if(grads.size() != weights.size())
                throw runtime_error("Input grads size does not match no. of neurons for layer: "+this->label+".");

            vector<double> op = vector<double>(weights[0].size(), 0);
            for(int i=0 ; i<weights.size() ; i++){
                for(int j=0 ; j<weights[i].size() ; j++){
                    weights[i][j].grad += input[j] * grads[i];
                    op[j] += weights[i][j].val * grads[i];
                }
            }

            // cout<<this->label<<":[ ";
            // for(double d: op)
            //     cout<<d<<", ";
            // cout<<"\b\b ]\n";

            return op;
        }

        void update_weights(double learning_rate = 0.1){
            for(int i=0 ; i<weights.size() ; i++){
                for(int j=0 ; j<weights[i].size() ; j++){
                    weights[i][j].val -= learning_rate * weights[i][j].grad;
                    weights[i][j].grad = 0;
                }
            }
        }
};

class Conv2d : public Layer{
    public:
        vector<vector<vector<vector<Value>>>> filters;
        Tensor3D input;
        int stride;
        int padding;
        int filter_size;
        int dilation;
        tuple<int,int,int> input_dim;
        tuple<int,int,int> output_dim;

        Conv2d(){}

        Conv2d(
            string label,
            tuple<int,int,int> input_dim,
            int filter_size,
            int no_of_filters,
            int stride=1,
            int padding=0,
            int dilation=1
        ){
            if(stride <= 0)
                throw runtime_error("Stride cannot be 0 or negative for Conv2d layer: "+label);
            if(padding < 0)
                throw runtime_error("Padding cannot be negative for Conv2d layer: "+label);
            if(get<0>(input_dim)<=0 || get<1>(input_dim)<=0 || get<2>(input_dim)<=0)
                throw runtime_error("Input dimension cannot be less than 0 for Conv2d layer: "+label);

            this->label = label;
            this->input_dim = input_dim;
            int effective_filter_size = dilation * (filter_size - 1) + 1;
            this->output_dim = {
                ((get<0>(input_dim) + 2*padding - effective_filter_size) / stride + 1),
                ((get<1>(input_dim) + 2*padding - effective_filter_size) / stride + 1),
                no_of_filters
            };
            this->filters = vector<vector<vector<vector<Value>>>>(no_of_filters, vector<vector<vector<Value>>>(get<2>(input_dim), vector<vector<Value>>(filter_size, vector<Value>(filter_size))));
            this->input = vector<vector<vector<double>>>(get<2>(input_dim), vector<vector<double>>(get<0>(input_dim), vector<double>(get<1>(input_dim))));
            this->stride = stride;
            this->filter_size = filter_size;
            this->padding = padding;
            this->dilation = dilation;
            for(int i=0 ; i < filters.size() ; i++){
                for(int j=0 ; j < filters[i].size() ; j++){
                    for(int k=0 ; k < filters[i][j].size() ; k++){
                        for(int l=0 ; l < filters[i][j][k].size() ; l++){
                            filters[i][j][k][l] = Value(
                                "Conv2d: "+this->label+" param("+to_string(i)+','+to_string(j)+','+to_string(k)+','+to_string(l)+").",
                                generate_random_in_range(-0.1, 0.1)
                            );
                        }
                    }
                }
            }
        }

        
        template <typename T>
        std::enable_if_t<
            std::is_same_v<T, double> || 
            std::is_same_v<T, Value>,
            vector<vector<double>>
        >
        convolve(
            vector<vector<double>> mat, 
            vector<vector<T>> filter, 
            int padding = 0, 
            int stride = 1, 
            int dilation = 1
        ){
            // 1. Get dimensions
            int input_H = mat.size();
            int input_W = mat[0].size();
            int filter_H = filter.size();
            int filter_W = filter[0].size();

            // 2. Calculate output dimensions (standard formula)
            // H_out = floor((H_in - H_filter + 2*P) / S) + 1
            // Note: Dilation is already accounted for in the H_filter term (H_filter_effective = D * (H_filter - 1) + 1)
            // For non-unit dilation, the formula should use the effective filter size.
            int effective_filter_H = dilation * (filter_H - 1) + 1;
            int effective_filter_W = dilation * (filter_W - 1) + 1;

            int out_H = (input_H - effective_filter_H + 2 * padding) / stride + 1;
            int out_W = (input_W - effective_filter_W + 2 * padding) / stride + 1;

            // Check for invalid dimension, though integer division in C++ often handles this.
            if (out_H <= 0 || out_W <= 0) {
                return {}; // Return empty matrix if output dimensions are invalid
            }

            // 3. Initialize the output matrix (Height, then Width)
            vector<vector<double>> ans(out_H, vector<double>(out_W));

            // 4. Perform Convolution
            // out_y = Row (Height), out_x = Column (Width)
            for (int out_y = 0; out_y < out_H; out_y++) {
                for (int out_x = 0; out_x < out_W; out_x++) {
                    double sum = 0;
                    
                    // Loop over filter rows (i) and columns (j)
                    for (int i = 0; i < filter_H; i++) {
                        for (int j = 0; j < filter_W; j++) {
                            
                            // Calculate corresponding input coordinates (y, x)
                            // in_coord = out_coord * Stride + Dilation * Filter_coord - Padding
                            int in_y = out_y * stride + dilation * i - padding;
                            int in_x = out_x * stride + dilation * j - padding;
                            
                            // Boundary Check
                            if (in_y >= 0 && in_y < input_H && 
                                in_x >= 0 && in_x < input_W) 
                            {
                                // Standard Convolution: Filter element * Input element
                                // Note the correct [row/y][column/x] indexing for mat
                                if constexpr(std::is_same_v<T, Value>)
                                    sum += filter[i][j].val * mat[in_y][in_x];                                    
                                else
                                    sum += filter[i][j] * mat[in_y][in_x];
                            }
                            // If out of bounds, padding assumes a zero, so we do nothing.
                        }
                    }
                    ans[out_y][out_x] = sum;
                }
            }

            return ans;
        }

        Tensor forward(Tensor t_input){
            if(!holds_alternative<Tensor3D>(t_input))
                throw runtime_error("Invalid input Tensor to Conv2d layer: "+this->label+".");
            
            const Tensor3D& input = std::get<Tensor3D>(t_input);
            if(input.size() != get<2>(input_dim))
                throw runtime_error("Invalid input dimensions for layer: "+this->label);
            
            vector<vector<vector<double>>> ans = vector<vector<vector<double>>>(get<2>(output_dim), vector<vector<double>>(get<0>(output_dim), vector<double>(get<1>(output_dim),0)));
            for (int oc = 0; oc < get<2>(output_dim); oc++) {
                vector<vector<double>> ans_oc(get<0>(output_dim), vector<double>(get<1>(output_dim), 0));
                for (int ic = 0; ic < get<2>(input_dim); ic++) {
                    ans_oc = ans_oc + convolve(input[ic], filters[oc][ic], padding, stride, dilation);
                }
                ans[oc] = ans_oc;
            }

            this -> input = input;

            return ans;
        }
        
        Tensor back_prop(Tensor t_grads){
            if(!holds_alternative<Tensor3D>(t_grads))
                throw runtime_error("Invalid gradient Tensor to Conv2d layer: "+this->label+".");
            
            const Tensor3D& grads = std::get<Tensor3D>(t_grads);
            if(grads.size() != get<2>(output_dim))
                throw runtime_error("Invalid no. of channels in output grads for layer: "+this->label);

            vector<vector<vector<double>>> ans(get<2>(input_dim), vector<vector<double>>(get<0>(input_dim), vector<double>(get<1>(input_dim), 0)));
            for (int oc = 0; oc < get<2>(output_dim); oc++) {
                if(grads[oc].size() != get<0>(output_dim))
                    throw runtime_error("Invalid x-dim in output grads for layer: "+this->label);
                for (int out_x = 0; out_x < get<0>(output_dim); out_x++) {
                    if(grads[oc][out_x].size() != get<1>(output_dim))
                        throw runtime_error("Invalid y-dim in output grads for layer: "+this->label);
                    for (int out_y = 0; out_y < get<1>(output_dim); out_y++) {
                        for (int ic = 0; ic < get<2>(input_dim); ic++) {
                            for (int i = 0; i < filter_size; i++) {
                                for (int j = 0; j < filter_size; j++) {
                                    int in_x = out_x * stride + dilation * i - padding;
                                    int in_y = out_y * stride + dilation * j - padding;
                                    if (in_x >= 0 && in_x < get<0>(input_dim) && in_y >= 0 && in_y < get<1>(input_dim)) {
                                        ans[ic][in_x][in_y] += filters[oc][ic][i][j].val * grads[oc][out_x][out_y];
                                        filters[oc][ic][i][j].grad += input[ic][in_x][in_y] * grads[oc][out_x][out_y];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return ans;
        }

        void update_weights(double learning_rate = 0.1){
            for(int i=0 ; i < filters.size() ; i++){
                for(int j=0 ; j < filters[i].size() ; j++){
                    for(int k=0 ; k < filters[i][j].size() ; k++){
                        for(int l=0 ; l < filters[i][j][k].size() ; l++){
                            filters[i][j][k][l].val -= learning_rate * filters[i][j][k][l].grad;
                            filters[i][j][k][l].grad = 0;
                        }
                    }
                }
            }
        }
};

class Flatten : public Layer{
    public:
        std::tuple<int, int, int> input_shape;

        Flatten(){}

        Flatten(string label){
            this->label = label;
        }

        Tensor forward(Tensor t_input) {
            if(holds_alternative<Tensor1D>(t_input))
                throw runtime_error("Invalid input Tensor to Flatten layer: "+this->label+".");
            
            Tensor1D output;
            if(holds_alternative<Tensor2D>(t_input)){
                const Tensor2D& input = std::get<Tensor2D>(t_input);
                this->input_shape = {input.size(), input[0].size(), 0};
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

        Tensor back_prop(Tensor t_grads) {
            if(!holds_alternative<Tensor1D>(t_grads))
                throw runtime_error("Invalid gradient Tensor to Flatten layer: "+this->label+".");

            Tensor1D& grads = std::get<Tensor1D>(t_grads);
            int x=get<0>(input_shape);
            int y=get<1>(input_shape);
            int z=get<2>(input_shape);

            if(z == 0){
                Tensor2D ip_grads = Tensor2D(x, Tensor1D(y));
                for(int i=0 ; i<x ; i++)
                    for(int j=0 ; j<y ; j++)
                        ip_grads[i][j] = grads[i*y + j];
                return ip_grads;
            }
            else{
                Tensor3D ip_grads = Tensor3D(x, Tensor2D(y, Tensor1D(z)));
                for(int i=0 ; i<x ; i++)
                    for(int j=0 ; j<y ; j++)
                        for(int k=0 ; k<z ; k++)
                            ip_grads[i][j][k] = grads[i*z*y + j*z + k];
                return ip_grads;
            }
        }

        void update_weights(double lr = 0.1) {
            // Flatten has no weights, so this is correct.
        }
};

class Softmax : public Layer{
    public: 
        Tensor1D input;

        Softmax(){}

        Softmax(string label){
            this->label = label;
        }

        Tensor forward(const Tensor t_input){
            if(!holds_alternative<Tensor1D>(t_input))
                throw runtime_error("Invalid input Tensor to Softmax layer: "+this->label+".");

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

        Tensor back_prop(const Tensor t_grads){
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

        void update_weights(double lr = 0.1){}
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Gemini-2.5-pro version4903
class ReLU : public Layer {
public:

    // The cache for the input from the forward pass
    Tensor input_cache;

    ReLU() {}
    ReLU(std::string label) {
        this->label = label;
    }

    Tensor forward(Tensor t_input) {
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

    Tensor back_prop(Tensor t_grads) {
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

    void update_weights(double lr = 0.1) {
        // ReLU has no weights, so this is correct.
    }
};