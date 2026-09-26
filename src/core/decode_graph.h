#pragma once

#include <cuda_runtime.h>

#include <functional>

namespace ninfer {

namespace detail {
template<class Query,class Drain,class Retire>
cudaError_t retire_decode_graph_device(int expected,Query&& query,Drain&& drain,Retire&& retire) noexcept {
    if(expected<0)return cudaErrorInvalidDevice;
    int current=-1;
    auto error=query(&current);
    if(error!=cudaSuccess)return error;
    if(current!=expected)return cudaErrorInvalidDevice;
    error=drain();
    return error==cudaSuccess?retire():error;
}
template<class Executable,class Definition>
cudaError_t retire_decode_graph_pair(Executable& executable,Definition& definition) noexcept {
    const auto error=executable.try_reset();
    return error==cudaSuccess?definition.try_reset():error;
}
template<class Handle,class Destroy>
cudaError_t retire_decode_graph_handle(Handle& handle,Destroy&& destroy) noexcept {
    if(!handle)return cudaSuccess;
    const auto error=destroy(handle);
    if(error==cudaSuccess)handle=nullptr;
    return error;
}
template<class Handle,class Destroy>
cudaError_t retire_decode_graph_handle_once(Handle& handle,cudaError_t& first_error,Destroy&& destroy) noexcept {
    if(first_error!=cudaSuccess)return first_error;
    first_error=retire_decode_graph_handle(handle,destroy);
    return first_error;
}
}

class DecodeGraphDefinition {
public:
    struct Provider {
        cudaError_t (*begin)(cudaStream_t)=+[](cudaStream_t stream){return cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);};
        cudaError_t (*end)(cudaStream_t,cudaGraph_t*)=+[](cudaStream_t stream,cudaGraph_t* graph){return cudaStreamEndCapture(stream,graph);};
        cudaError_t (*destroy)(cudaGraph_t)=+[](cudaGraph_t graph){return cudaGraphDestroy(graph);};
    };
    DecodeGraphDefinition() = default;
    explicit DecodeGraphDefinition(Provider provider);
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    // Failed destination retirement preserves both handles/providers. Call
    // try_reset explicitly when the caller needs to report that failure.
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(cudaStream_t stream, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    // Caller proves final use. Failed destruction preserves the handle so its
    // enclosing resource owner can be quarantined without losing provenance.
    [[nodiscard]] cudaError_t try_reset() noexcept;
    void reset() noexcept;

private:
    friend class DecodeGraphExecutable;
    cudaGraph_t graph_ = nullptr;
    cudaError_t retirement_error_ = cudaSuccess;
    Provider provider_;
};

class DecodeGraphExecutable {
public:
    struct Provider {
        cudaError_t (*instantiate)(cudaGraphExec_t*,cudaGraph_t)=+[](cudaGraphExec_t* exec,cudaGraph_t graph){return cudaGraphInstantiate(exec,graph,0);};
        cudaError_t (*destroy)(cudaGraphExec_t)=+[](cudaGraphExec_t exec){return cudaGraphExecDestroy(exec);};
        cudaError_t (*upload)(cudaGraphExec_t,cudaStream_t)=+[](cudaGraphExec_t exec,cudaStream_t stream){return cudaGraphUpload(exec,stream);};
        cudaError_t (*launch)(cudaGraphExec_t,cudaStream_t)=+[](cudaGraphExec_t exec,cudaStream_t stream){return cudaGraphLaunch(exec,stream);};
    };
    DecodeGraphExecutable() = default;
    explicit DecodeGraphExecutable(Provider provider);
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    // Failed destination retirement preserves both handles/providers.
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(cudaStream_t stream);
    void launch(cudaStream_t stream);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] cudaError_t try_reset() noexcept;
    void reset() noexcept;

private:
    cudaGraphExec_t exec_ = nullptr;
    cudaError_t retirement_error_ = cudaSuccess;
    Provider provider_;
    bool custom_provider_ = false;
};

} // namespace ninfer
