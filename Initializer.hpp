#pragma once
#include <typeindex> // Needed for using type_info as a map key
#include "Layers.hpp"

// The interface for our initialization strategies
class InitializerStrategy {
public:
    virtual ~InitializerStrategy() = default;

    // The method to be overridden. It takes the layer to be initialized
    // and the previous layer (which is already initialized).
    virtual void initialize(Layer* current_layer, const Layer* previous_layer) const = 0;
};

class Initializer {
private:
    // The registry mapping a Layer's type_index to its initialization strategy
    std::unordered_map<std::type_index, std::unique_ptr<InitializerStrategy>> strategies;

public:
    void show_strategies(){
        for(auto it = strategies.begin() ; it != strategies.end() ; it++){
            cout<<it->first.name()<<endl;
        }
    }

    // Method to register a strategy for a specific Layer type
    template <typename T>
    void register_strategy(std::unique_ptr<InitializerStrategy> strategy) {
        strategies[std::type_index(typeid(T))] = std::move(strategy);
    }

    // The main method that configures the whole network
    void compile(std::vector<std::unique_ptr<Layer>>& network, Shape first_layer_input_shape) {
        if (network.empty()) return;

        // 1. Manually initialize the first layer
        // cout<<"Initializing layer: 0"<<endl;
        network[0]->input_shape = first_layer_input_shape;
        network[0]->compute_output_shape();

        // 2. Loop through the rest of the layers
        for (size_t i = 1; i < network.size(); ++i) {
            Layer* previous = network[i - 1].get();
            Layer* current = network[i].get();
            // cout<<"Initializing layer: "<<i<<", Previous output shape: "<<previous->output_shape<<endl;

            // Find the strategy for the current layer's type
            auto it = strategies.find(std::type_index(typeid(*current)));
            if (it == strategies.end()) {
                throw std::runtime_error("No initializer strategy found for layer type: " + std::string(typeid(*current).name()));
            }
            // cout<<"Strategy found: "<<it->first.name()<<endl;

            // Execute the strategy
            it->second->initialize(current, previous);
        }
    }
};

// Strategy for a Dense layer
class DenseInitializer : public InitializerStrategy {
public:
    void initialize(Layer* current_layer, const Layer* previous_layer) const override {
        // Safely downcast the pointer to the concrete Dense type
        Dense* dense_layer = dynamic_cast<Dense*>(current_layer);
        if (!dense_layer) return; // Or throw an error

        // The magic happens here:
        // 1. Get the previous layer's output shape
        const Shape& prev_output_shape = previous_layer->output_shape;

        if(prev_output_shape.channels != 0 || prev_output_shape.height != 0)
            throw invalid_argument("Initializer error: Input shape to a Dense layer must be linear. for layer: "+current_layer->label+".");

        // 2. Set the current layer's input shape
        dense_layer->input_shape = prev_output_shape;
        
        // 3. (Important!) Tell the layer to initialize its internal weights
        //    and compute its own output shape based on the new input shape.
        dense_layer->compute_output_shape();
    }
};

// Strategy for a Conv2D layer
class Conv2DInitializer : public InitializerStrategy {
public:
    void initialize(Layer* current_layer, const Layer* previous_layer) const override {
        // Safely downcast the pointer to the concrete Conv2D type
        Conv2D* conv2d_layer = dynamic_cast<Conv2D*>(current_layer);
        if (!conv2d_layer) return; // Or throw an error

        // The magic happens here:
        // 1. Get the previous layer's output shape
        const Shape& prev_output_shape = previous_layer->output_shape;

        if(prev_output_shape.channels == 0 || prev_output_shape.height == 0)
            throw invalid_argument("Initializer error: Input shape to a Conv2D layer must be a 3D Tensor. for layer: "+current_layer->label+".");

        // 2. Set the current layer's input shape
        conv2d_layer->input_shape = prev_output_shape;
        
        // 3. (Important!) Tell the layer to initialize its internal weights
        //    and compute its own output shape based on the new input shape.
        conv2d_layer->compute_output_shape();
    }
};

// Strategy for a MaxPool2D layer
class MaxPool2DInitializer : public InitializerStrategy {
public:
    void initialize(Layer* current_layer, const Layer* previous_layer) const override {
        // Safely downcast the pointer to the concrete MaxPool2D type
        MaxPool2D* pool_layer = dynamic_cast<MaxPool2D*>(current_layer);
        if (!pool_layer) return; // Or throw an error

        // The magic happens here:
        // 1. Get the previous layer's output shape
        const Shape& prev_output_shape = previous_layer->output_shape;

        if(prev_output_shape.channels == 0 || prev_output_shape.height == 0)
            throw invalid_argument("Initializer error: Input shape to a MaxPpol2D layer must be a 3D Tensor. for layer: "+current_layer->label+".");

        // 2. Set the current layer's input shape
        pool_layer->input_shape = prev_output_shape;
        
        // 3. (Important!) Tell the layer to initialize its internal weights
        //    and compute its own output shape based on the new input shape.
        pool_layer->compute_output_shape();
    }
};

// Strategy for a Flatten layer
class FlattenInitializer : public InitializerStrategy {
public:
    void initialize(Layer* current_layer, const Layer* previous_layer) const override {
        // Safely downcast the pointer to the concrete Flatten type
        Flatten* flatten_layer = dynamic_cast<Flatten*>(current_layer);
        if (!flatten_layer) return; // Or throw an error

        // The magic happens here:
        // 1. Get the previous layer's output shape
        const Shape& prev_output_shape = previous_layer->output_shape;

        // 2. Set the current layer's input shape
        flatten_layer->input_shape = prev_output_shape;
        
        // 3. (Important!) Tell the layer to initialize its internal weights
        //    and compute its own output shape based on the new input shape.
        flatten_layer->compute_output_shape();
    }
};

// Strategy for a ReLU layer
class ReLUInitializer : public InitializerStrategy {
public:
    void initialize(Layer* current_layer, const Layer* previous_layer) const override {
        // Safely downcast the pointer to the concrete ReLU type
        ReLU* relu_layer = dynamic_cast<ReLU*>(current_layer);
        if (!relu_layer) return; // Or throw an error

        // The magic happens here:
        // 1. Get the previous layer's output shape
        const Shape& prev_output_shape = previous_layer->output_shape;

        // 2. Set the current layer's input shape
        relu_layer->input_shape = prev_output_shape;
        
        // 3. (Important!) Tell the layer to initialize its internal weights
        //    and compute its own output shape based on the new input shape.
        relu_layer->compute_output_shape();
    }
};

// Strategy for a Softmax layer
class SoftmaxInitializer : public InitializerStrategy {
public:
    void initialize(Layer* current_layer, const Layer* previous_layer) const override {
        // Safely downcast the pointer to the concrete Softmax type
        Softmax* softmax_layer = dynamic_cast<Softmax*>(current_layer);
        if (!softmax_layer) return; // Or throw an error

        // The magic happens here:
        // 1. Get the previous layer's output shape
        const Shape& prev_output_shape = previous_layer->output_shape;

        if(prev_output_shape.channels != 0 || prev_output_shape.height != 0)
            throw invalid_argument("Initializer error: Input shape to a Dense layer must be linear. for layer: "+current_layer->label+".");

        // 2. Set the current layer's input shape
        softmax_layer->input_shape = prev_output_shape;
        
        // 3. (Important!) Tell the layer to initialize its internal weights
        //    and compute its own output shape based on the new input shape.
        softmax_layer->compute_output_shape();
    }
};

Initializer GlobalInitializerStrategies = [] {
    Initializer init; // Create a temporary Initializer

    // Perform all your actions on the temporary object
    init.register_strategy<Dense>(std::make_unique<DenseInitializer>());
    init.register_strategy<Conv2D>(std::make_unique<Conv2DInitializer>());
    init.register_strategy<Flatten>(std::make_unique<FlattenInitializer>());
    init.register_strategy<MaxPool2D>(std::make_unique<MaxPool2DInitializer>());
    init.register_strategy<ReLU>(std::make_unique<ReLUInitializer>());
    init.register_strategy<Softmax>(std::make_unique<SoftmaxInitializer>());
    // ... add more strategies ...

    // Return the fully configured object.
    // This will be "moved" into G_Initializer efficiently.
    return init; 
}(); // The () at the end immediately calls the lambda.


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