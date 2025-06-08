#include <torch/extension.h>
#include <vector>
#include <string>
#include <ATen/Parallel.h>
#include <ATen/Functions.h>


torch::Tensor newton_schulz5_single(const torch::Tensor& G, int steps, double eps) {

    const auto a = static_cast<float>(3.4445);
    const auto b = static_cast<float>(-4.7750);
    const auto c = static_cast<float>(2.0315);

    auto X = G.to(torch::kBFloat16);

    auto norm_X = at::linalg_norm(X) + eps;
    X = X / norm_X;

    bool needs_transpose = G.size(0) > G.size(1);
    if (needs_transpose) {
        X = at::transpose(X, 0, 1);
    }

    for (int i = 0; i < steps; ++i) {
        auto XT = at::transpose(X, 0, 1);
        auto A = at::matmul(X, XT);
        auto B = at::matmul(A, X);
        auto C_term = at::matmul(A, B);
        X = a * X + b * B + c * C_term;
    }

    if (needs_transpose) {
        X = at::transpose(X, 0, 1);
    }

    return X;
}


std::vector<torch::Tensor> process_gradients_parallel(
    std::vector<torch::Tensor>& grads,
    const std::string& backend,
    int backend_steps,
    double rms_norm_eps = 1e-8) {

    int64_t num_grads = grads.size();

    at::parallel_for(0, num_grads, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            auto& grad = grads[i];

            if (grad.dim() != 2) {
                 continue;
            }
            
            grad = newton_schulz5_single(grad, backend_steps, 1e-7);

            const int normalized_shape = grad.size(grad.dim() - 1);
            
            grads[i] = at::rms_norm(grad, normalized_shape, /*weight=*/c10::nullopt, /*eps=*/rms_norm_eps);

        }
    });

    return grads;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("process_gradients", &process_gradients_parallel, "Muon Gradient Processing Kernel (Orthogonalize + RMS Norm, Parallelized)");
} 
