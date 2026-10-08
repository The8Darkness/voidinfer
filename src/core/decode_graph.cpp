#include "core/decode_graph.h"

#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

} // namespace

DecodeGraphDefinition::~DecodeGraphDefinition() { reset(); }
DecodeGraphDefinition::DecodeGraphDefinition(Provider provider):provider_(provider) {
    if(!provider_.begin || !provider_.end || !provider_.destroy)
        throw std::invalid_argument("incomplete graph definition provider");
}

DecodeGraphDefinition::DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept
    : graph_(other.graph_),retirement_error_(other.retirement_error_),provider_(other.provider_) {
    other.graph_ = nullptr;
    other.retirement_error_=cudaSuccess;
}

DecodeGraphDefinition& DecodeGraphDefinition::operator=(DecodeGraphDefinition&& other) noexcept {
    if (this == &other) { return *this; }

    if(try_reset()!=cudaSuccess)return *this;
    graph_ = other.graph_;
    provider_=other.provider_;
    retirement_error_=other.retirement_error_;

    other.graph_ = nullptr;
    other.retirement_error_=cudaSuccess;
    return *this;
}

void DecodeGraphDefinition::capture(cudaStream_t stream, const std::function<void()>& body) {
    CUDA_CHECK(try_reset());

    CUDA_CHECK(provider_.begin(stream));

    try {
        body();
    } catch (...) {
        log_cuda_error("cudaStreamEndCapture(discard)",provider_.end(stream,&graph_));
        (void)try_reset(); // Preserve a failed destroy and the original body exception.
        throw;
    }

    cudaError_t err = provider_.end(stream, &graph_);
    if (err != cudaSuccess) {
        (void)try_reset();
        CUDA_CHECK(err);
    }
}

bool DecodeGraphDefinition::ready() const noexcept { return graph_ != nullptr; }

cudaError_t DecodeGraphDefinition::try_reset() noexcept {
    return detail::retire_decode_graph_handle_once(graph_,retirement_error_,provider_.destroy);
}

void DecodeGraphDefinition::reset() noexcept { log_cuda_error("cudaGraphDestroy",try_reset()); }

DecodeGraphExecutable::~DecodeGraphExecutable() { reset(); }
DecodeGraphExecutable::DecodeGraphExecutable(Provider provider)
    :provider_(provider),custom_provider_(true) {
    if(!provider_.instantiate || !provider_.destroy || !provider_.upload ||
       !provider_.launch)
        throw std::invalid_argument("incomplete graph executable provider");
}

DecodeGraphExecutable::DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept
    : exec_(other.exec_),retirement_error_(other.retirement_error_),
      provider_(other.provider_),custom_provider_(other.custom_provider_) {
    other.exec_ = nullptr;
    other.retirement_error_=cudaSuccess;
    other.custom_provider_=false;
}

DecodeGraphExecutable& DecodeGraphExecutable::operator=(DecodeGraphExecutable&& other) noexcept {
    if (this == &other) { return *this; }

    if(try_reset()!=cudaSuccess)return *this;
    exec_       = other.exec_;
    provider_=other.provider_;
    custom_provider_=other.custom_provider_;
    retirement_error_=other.retirement_error_;
    other.exec_ = nullptr;
    other.retirement_error_=cudaSuccess;
    other.custom_provider_=false;
    return *this;
}

void DecodeGraphExecutable::instantiate(const DecodeGraphDefinition& definition) {
    CUDA_CHECK(definition.retirement_error_);
    if (!definition.ready()) {
        throw std::logic_error("cannot instantiate an empty CUDA Graph definition");
    }
    CUDA_CHECK(try_reset());

    const cudaError_t err = provider_.instantiate(&exec_, definition.graph_);
    if (err != cudaSuccess) {
        (void)try_reset();
        CUDA_CHECK(err);
    }
}

void DecodeGraphExecutable::update(const DecodeGraphDefinition& definition) {
    CUDA_CHECK(retirement_error_);
    CUDA_CHECK(definition.retirement_error_);
    if (!ready() || !definition.ready()) {
        throw std::logic_error("CUDA Graph update requires a definition and executable");
    }

    cudaGraphExecUpdateResultInfo result{};
    const cudaError_t err = cudaGraphExecUpdate(exec_, definition.graph_, &result);
    if (err != cudaSuccess || result.result != cudaGraphExecUpdateSuccess) {
        throw std::runtime_error(
            "CUDA Graph executable update failed: " + std::string(cudaGetErrorName(err)) +
            " (update result " + std::to_string(static_cast<int>(result.result)) + ")");
    }
}

void DecodeGraphExecutable::upload(cudaStream_t stream) {
    CUDA_CHECK(retirement_error_);
    if (!ready()) { throw std::logic_error("cannot upload an empty CUDA Graph executable"); }
    // The default upload and launch callback pointers aliased in an
    // Engine-linked Windows binary. Dispatch ordinary CUDA calls directly;
    // explicit providers remain available for failure injection.
    CUDA_CHECK(custom_provider_?provider_.upload(exec_,stream):
        cudaGraphUpload(exec_,stream));
}

void DecodeGraphExecutable::launch(cudaStream_t stream) {
    CUDA_CHECK(retirement_error_);
    if (!ready()) { throw std::logic_error("cannot launch an empty CUDA Graph executable"); }
    CUDA_CHECK(custom_provider_?provider_.launch(exec_,stream):
        cudaGraphLaunch(exec_,stream));
}

bool DecodeGraphExecutable::ready() const noexcept { return exec_ != nullptr; }

cudaError_t DecodeGraphExecutable::try_reset() noexcept {
    return detail::retire_decode_graph_handle_once(exec_,retirement_error_,provider_.destroy);
}

void DecodeGraphExecutable::reset() noexcept { log_cuda_error("cudaGraphExecDestroy",try_reset()); }

} // namespace ninfer
