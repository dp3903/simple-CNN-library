#include "Model.hpp"
#include "Layer.hpp"
#include "Dense.hpp"


// ===== Helper functions to avoid writing make_unique every time =====
std::unique_ptr<Dense> DenseLayer(const std::string& name, int in_f, int out_f) {
    return std::make_unique<Dense>(name, in_f, out_f);
}