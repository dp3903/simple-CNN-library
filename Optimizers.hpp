#include "Layers.hpp"

class Optimizer {
public:
    // Store references to the layers, not copies
    vector<unique_ptr<Layer>>& layers;
    double learning_rate;

    Optimizer(vector<unique_ptr<Layer>>& model_layers, double lr)
        : layers(model_layers), learning_rate(lr) {}

    virtual ~Optimizer() = default;

    // The core function that every optimizer must implement
    virtual void step() = 0;

    void zero_grad(){
        // Loop through each layer in the model
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i]->trainable && layers[i]->parameter_count() > 0) {
                // You define the update logic in a lambda function
                auto zero_rule = [&](Value& param, int& index) {
                    param.grad = 0;
                };

                // You then pass this update rule to the layer
                layers[i]->update_weights(zero_rule);
            }
        }
    };
};

class SGD : public Optimizer {
    public:
        SGD(vector<unique_ptr<Layer>>& model_layers, double lr = 0.01)
            : Optimizer(model_layers, lr)
        {}

        void step() override {
            // Loop through each layer in the model
            for (size_t i = 0; i < layers.size(); ++i) {
                if (layers[i]->trainable && layers[i]->parameter_count() > 0) {
                    // In your Optimizer::step() method
                    // You define the update logic in a lambda function
                    auto update_rule = [&](Value& param, int& index) {
                        param.val -= learning_rate * param.grad;
                    };

                    // You then pass this update rule to the layer
                    layers[i]->update_weights(update_rule);
                }
            }
        }
};

class MomentumSGD : public Optimizer {
    private:
        double beta;
        vector<Tensor1D> velocities;

    public:
        MomentumSGD(vector<unique_ptr<Layer>>& model_layers, double lr = 0.01, double beta = 0.9)
            : Optimizer(model_layers, lr), beta(beta)
        {
            velocities = vector<Tensor1D>(layers.size());
            for(size_t i=0 ; i < layers.size() ; i++)
                if(layers[i]->trainable && layers[i]->parameter_count() > 0)
                    velocities[i] = Tensor1D(layers[i]->parameter_count(), 0);
        }

        void step() override {
            // Loop through each layer in the model
            for (size_t i = 0; i < layers.size(); i++) {
                if (layers[i]->trainable && layers[i]->parameter_count() > 0) {
                    if(layers[i]->parameter_count() != velocities[i].size())
                        throw runtime_error("Optimizer velocities count does not match parameter count for layer:'"+layers[i]->label+"'.");
                    // In your Optimizer::step() method
                    // You define the update logic in a lambda function
                    auto update_rule = [&](Value& param, int& index) {
                        if(index < 0 || index >= velocities[i].size())
                            throw runtime_error("Invalid parameter index passed in the optimizer.step for layer:'"+layers[i]->label+"'.");

                        velocities[i][index] = beta*velocities[i][index] + (1-beta)*param.grad;
                        param.val -= learning_rate * velocities[i][index];
                    };

                    // You then pass this update rule to the layer
                    layers[i]->update_weights(update_rule);
                }
            }
        }
};

class Adam : public Optimizer {
    private:
        double beta1;
        double beta2;
        double epsilon;
        int timestamp = 0;
        vector<vector<pair<double,double>>> moments;

    public:
        Adam(vector<unique_ptr<Layer>>& model_layers, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1.0e-8)
            : Optimizer(model_layers, lr), beta1(beta1), beta2(beta2), epsilon(epsilon)
        {
            moments = vector<vector<pair<double,double>>>(layers.size());
            for(size_t i=0 ; i < layers.size() ; i++)
                if(layers[i]->trainable && layers[i]->parameter_count() > 0)
                    moments[i] = vector<pair<double,double>>(layers[i]->parameter_count(), {0, 0});
        }

        void step() override {
            // update timestamp
            timestamp++;
            
            // Loop through each layer in the model
            for (size_t i = 0; i < layers.size(); i++) {
                if (layers[i]->trainable && layers[i]->parameter_count() > 0) {
                    if(layers[i]->parameter_count() != moments[i].size())
                        throw runtime_error("Optimizer moments' count does not match parameter count for layer:'"+layers[i]->label+"'.");
                    // In your Optimizer::step() method
                    // You define the update logic in a lambda function
                    auto update_rule = [&](Value& param, int& index) {
                        if(index < 0 || index >= moments[i].size())
                            throw runtime_error("Invalid parameter index passed in the optimizer.step for layer:'"+layers[i]->label+"'.");

                        // update mt and vt
                        moments[i][index].first = beta1*(moments[i][index].first) + (1-beta1)*param.grad;
                        moments[i][index].second = beta2*(moments[i][index].second) + (1-beta2)*param.grad*param.grad;

                        // correct for bias of the moments initialized to 0
                        double c_mt = moments[i][index].first / (1 - pow(beta1, timestamp));
                        double c_vt = moments[i][index].second / (1 - pow(beta2, timestamp));

                        // update params
                        param.val -= learning_rate * c_mt / (sqrt(c_vt) + epsilon);
                    };

                    // You then pass this update rule to the layer
                    layers[i]->update_weights(update_rule);
                }
            }
        }
};