/**
 * Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
 *
 * This file is part of the ParaS Compiler, a component of the ParaS Ecosystem.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARAS_GPU_THREADPOOL_HPP__
#define __PARAS_GPU_THREADPOOL_HPP__

#include "sycl/context.hpp"
#include "sycl/device.hpp"
#include "sycl/device_selector.hpp"
#include "sycl/event.hpp"
#include "sycl/exception.hpp"
#include "kem_gpu/offload_arch_check.hpp"
#include "sycl/id.hpp"
#include "sycl/interop_handle.hpp"
#include "sycl/item.hpp"
#include "sycl/queue.hpp"
#include "sycl/range.hpp"
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <iostream>

#include "sycl/handler.hpp"
#include "sycl/queue.hpp"
#include "utilities/selector_logic.hpp"

#include "gpu_utilities.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

class paras_shared_flag {
public:
    operator bool() const noexcept { return *value_; }
    paras_shared_flag& operator=(bool v) noexcept {
        *value_ = v;
        return *this;
    }

private:
    std::shared_ptr<bool> value_ = std::make_shared<bool>(false);
};

struct paras_record_state {
    std::string error;
    std::vector<std::shared_ptr<void>> host_tasks;
};

#if (PARAS_CUDA_BACKEND)
#include <cuda_runtime.h>


class cuda_threadpool {

public:
    sycl::backend get_backend() const {
#if PARAS_GPU_BACKEND
        return sycl::backend::cuda;
#else
        return sycl::backend::host;
#endif
    }

    sycl::device get_device() const { return dev_; }
    sycl::context get_context() const { return ctx_; }

    cudaStream_t stream = nullptr;
    bool ownsStream = false;

    cuda_threadpool() : dev_(::paras_extension::select_device_no_selector()) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    explicit cuda_threadpool(const Selector& sel, const sycl::property_list& props = {})
        : dev_(::paras_extension::select_device_with_selector(sel)), props_(props) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    cuda_threadpool(const Selector& sel, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : cuda_threadpool(sel, props) {}

    explicit cuda_threadpool(const sycl::property_list& props)
        : dev_(::paras_extension::select_device_no_selector()), props_(props) {
        initialize_stream_for_device();
    }

    explicit cuda_threadpool(const sycl::async_handler&, const sycl::property_list& props = {})
        : cuda_threadpool(props) {}

    cuda_threadpool(const sycl::device& dev, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : cuda_threadpool(dev, props) {}

    bool operator==(const cuda_threadpool& rhs) const noexcept { return id_ == rhs.id_; }
    bool operator!=(const cuda_threadpool& rhs) const noexcept { return !(*this == rhs); }

    cuda_threadpool(const sycl::queue& q) : dev_(q.get_device()), ctx_(q.get_context()), queue_(q) {
        queue_.paras_detach_pool();
        initialize_stream_for_device();
    }

    explicit cuda_threadpool(const sycl::context& ctx, const sycl::device& dev,
                             const sycl::property_list& props = {})
        : dev_(dev), ctx_(ctx), props_(props) {
        initialize_stream_for_device();
    }

    explicit cuda_threadpool(const sycl::device& dev, const sycl::property_list& props = {})
        : dev_(dev), props_(props) {
        initialize_stream_for_device();
    }

    cuda_threadpool(const cuda_threadpool& other)
        : stream(other.stream), ownsStream(other.ownsStream), dev_(other.dev_), ctx_(other.ctx_),
          props_(other.props_), queue_(other.queue_), id_(other.id_),
          stream_alive_(other.stream_alive_), stream_owner_(other.stream_owner_),
          recording_(other.recording_), record_(other.record_) {}

    cuda_threadpool& operator=(const cuda_threadpool& other) {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        dev_ = other.dev_;
        ctx_ = other.ctx_;
        props_ = other.props_;
        queue_ = other.queue_;
        id_ = other.id_;
        stream = other.stream;
        ownsStream = other.ownsStream;
        stream_alive_ = other.stream_alive_;
        stream_owner_ = other.stream_owner_;
        recording_ = other.recording_;
        record_ = other.record_;
        return *this;
    }

    cuda_threadpool(cuda_threadpool&& other) noexcept
        : stream(other.stream), ownsStream(other.ownsStream), dev_(std::move(other.dev_)),
          ctx_(std::move(other.ctx_)), props_(std::move(other.props_)),
          queue_(std::move(other.queue_)), id_(other.id_),
          stream_alive_(std::move(other.stream_alive_)), stream_owner_(std::move(other.stream_owner_)),
          recording_(other.recording_), record_(other.record_) {
        other.stream = nullptr;
        other.ownsStream = false;
    }

    cuda_threadpool& operator=(cuda_threadpool&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        stream = other.stream;
        ownsStream = other.ownsStream;
        stream_alive_ = std::move(other.stream_alive_);
        stream_owner_ = std::move(other.stream_owner_);
        recording_ = other.recording_;
        record_ = other.record_;
        dev_ = std::move(other.dev_);
        ctx_ = std::move(other.ctx_);
        props_ = std::move(other.props_);
        queue_ = std::move(other.queue_);
        id_ = other.id_;

        other.stream = nullptr;
        other.ownsStream = false;

        return *this;
    }

    std::uint64_t paras_profiling_start() const {
        return props_.has_profiling() ? sycl::paras_now_ns() : 0;
    }
    sycl::event paras_stream_event() const {
        cudaStream_t s = stream;
        std::shared_ptr<bool> alive = stream_alive_;
        return sycl::event::paras_async(
            sycl::backend::cuda,
            [s, alive] {
                if (!(alive && *alive)) return;
                if (stream_recording(s))
                    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                          "event::wait on a command recorded into a graph");
                const cudaError_t err = cudaStreamSynchronize(s);
                if (err != cudaSuccess)
                    throw sycl::exception(std::string("kernel execution failed: ") +
                                          cudaGetErrorString(err));
            },
            [s, alive] {
                return !(alive && *alive) ||
                       (!stream_recording(s) && cudaStreamQuery(s) != cudaErrorNotReady);
            });
    }
    sycl::event paras_finish_event(std::uint64_t t0) const {
        if (async_launch()) return paras_stream_event();
        if (!props_.has_profiling()) {
            return stream != nullptr ? paras_stream_event() : sycl::event{};
        }
        return sycl::event::paras_profiled(t0, t0, sycl::paras_now_ns());
    }

    bool paras_profiling() const noexcept { return props_.has_profiling(); }
    bool async_launch() const {
        return (async_launch_ || recording_) && !props_.has_profiling();
    }

    bool is_recording() const noexcept { return recording_; }

    void paras_record_host_task(std::function<void()> fn) {
        auto task = std::make_shared<std::function<void()>>(std::move(fn));
        const cudaError_t err = cudaLaunchHostFunc(
            stream, [](void* p) { (*static_cast<std::function<void()>*>(p))(); }, task.get());
        if (err != cudaSuccess) fail_recording("a host task (cudaLaunchHostFunc failed)");
        record_->host_tasks.push_back(std::move(task));
    }
    [[noreturn]] void paras_fail_recording(const char* what) const { fail_recording(what); }
    void paras_begin_recording();
    std::shared_ptr<void> paras_end_recording();
    std::shared_ptr<void> paras_instantiate_graph(void* graph);
    sycl::event paras_launch_graph(void* exec);
    sycl::ext::oneapi::experimental::queue_state ext_oneapi_get_state() const;
    sycl::event ext_oneapi_graph(const sycl::ext::oneapi::experimental::command_graph<
                                 sycl::ext::oneapi::experimental::graph_state::executable>& graph);

    const cudaDeviceProp& device_props() {
        if (!device_props_valid_) {
            const cudaError_t err = cudaGetDeviceProperties(&device_props_, dev_.get_native_id());
            if (err != cudaSuccess)
                throw sycl::exception(std::string("cudaGetDeviceProperties failed: ") +
                                      cudaGetErrorString(err));
            device_props_valid_ = true;
        }
        return device_props_;
    }
    bool is_in_order() const { return props_.has_in_order(); }
    const void* paras_identity() const noexcept { return id_.get(); }
    template <typename Property>
    bool has_property() const noexcept {
        if constexpr (std::is_same_v<Property, sycl::property::queue::in_order>) {
            return props_.has_in_order();
        } else if constexpr (std::is_same_v<Property, sycl::property::queue::enable_profiling>) {
            return props_.has_profiling();
        } else {
            return false;
        }
    }

    template <typename Func>
    void submit(Func cgf) {
        sycl::handler cgh(*this);
        cgf(cgh);
    }

    static unsigned gpu_get_num_threads();

    template <typename Func>
    sycl::event spawn_1D(Func f);

    template <typename Func>
    sycl::event spawn_1D_event(Func f);

    template <typename Func>
    void spawn_ND(Func f);

    void wait() {
        if (recording_) fail_recording("queue::wait");
        ensure_stream();
        cudaError_t errSync = cudaStreamSynchronize(stream);
        if (errSync != cudaSuccess) {
            throw sycl::exception(std::string("cudaStreamSynchronize in queue::wait failed: ") +
                                  cudaGetErrorString(errSync));
        }
    }

    void wait_and_throw() { wait(); }

    template <typename injectCustomFunc>
    sycl::event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
        sycl::handler cgh(*this);
        cgh.parasSYCL_enqueue_custom_operation(std::move(cgf));
        return paras_stream_event();
    }

    void wait() const {
        if (recording_) fail_recording("queue::wait");
        if (stream == nullptr) {
            throw sycl::exception("cuda_threadpool::wait called with null CUDA stream");
        }

        cudaError_t errSync = cudaStreamSynchronize(stream);
        if (errSync != cudaSuccess) {
            throw sycl::exception(std::string("cudaStreamSynchronize in queue::wait failed: ") +
                                  cudaGetErrorString(errSync));
        }
    }

    void wait_and_throw() const { wait(); }

    template <typename Func>
    void gpu_execute_1D(const sycl::range<1>& r, Func f);

    template <typename Func>
    void gpu_execute_1D_async(Func f);

    template <typename Func>
    void gpu_execute_2D(const sycl::range<2>& r, Func f);

    template <typename Func>
    void gpu_execute_3D(const sycl::range<3>& r, Func f);

    template <typename Func>
    void gpu_execute_nd_range_1D(const sycl::nd_range<1>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_2D(const sycl::nd_range<2>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_3D(const sycl::nd_range<3>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(sycl::range<dim> r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(const sycl::nd_range<dim>& r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_nd_range_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_nd_range_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_nd_range_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

#include "kem/queue_shortcuts.hpp"

    sycl::event memcpy(void* dest, const void* src, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (recording_ && !stream_ordered_copy(dest, src))
            fail_recording("a copy to or from host memory");

        cudaError_t err = cudaMemcpyAsync(dest, src, numBytes, cudaMemcpyDefault, stream);

        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") +
                                     cudaGetErrorString(err));
        }

        if (stream_ordered_copy(dest, src)) {
            return paras_finish_event(paras_t0);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize after memcpy failed: ") +
                                     cudaGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    template <typename T>
    sycl::event copy(const T* src, T* dest, size_t count) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (recording_ && !stream_ordered_copy(dest, src))
            fail_recording("a copy to or from host memory");

        cudaError_t err = cudaMemcpyAsync(static_cast<void*>(dest), static_cast<const void*>(src),
                                          count * sizeof(T), cudaMemcpyDefault, stream);

        if (err != cudaSuccess) {
            throw sycl::exception(cudaGetErrorString(err));
        }

        if (stream_ordered_copy(dest, src)) {
            return paras_finish_event(paras_t0);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw sycl::exception(std::string("cudaStreamSynchronize after copy failed: ") +
                                  cudaGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    sycl::event memset(void* ptr, int value, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        cudaError_t err = cudaMemsetAsync(ptr, value, numBytes, stream);

        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemsetAsync failed: ") +
                                     cudaGetErrorString(err));
        }

        if (async_launch()) {
            return paras_finish_event(paras_t0);
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaStreamSynchronize after memset failed: ") +
                                     cudaGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    sycl::event prefetch(const void* ptr, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (numBytes != 0 && !recording_) {
#if CUDART_VERSION >= 13000
            cudaMemLocation loc{};
            loc.type = cudaMemLocationTypeDevice;
            loc.id = dev_.get_native_id();
            const cudaError_t err = cudaMemPrefetchAsync(ptr, numBytes, loc, 0, stream);
#else
            const cudaError_t err = cudaMemPrefetchAsync(ptr, numBytes, dev_.get_native_id(), stream);
#endif
            if (err != cudaSuccess) (void)cudaGetLastError();
        }
        if (!async_launch()) (void)cudaStreamSynchronize(stream);
        return paras_finish_event(paras_t0);
    }

    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (numBytes != 0 && !recording_) {
#if CUDART_VERSION >= 13000
            cudaMemLocation loc{};
            loc.type = cudaMemLocationTypeDevice;
            loc.id = dev_.get_native_id();
            const cudaError_t err = cudaMemAdvise(ptr, numBytes, static_cast<cudaMemoryAdvise>(advice), loc);
#else
            const cudaError_t err =
                cudaMemAdvise(ptr, numBytes, static_cast<cudaMemoryAdvise>(advice), dev_.get_native_id());
#endif
            if (err != cudaSuccess) (void)cudaGetLastError();
        }
        return paras_finish_event(paras_t0);
    }

    sycl::event prefetch(const void* ptr, size_t numBytes, sycl::event dep) {
        dep.wait();
        return prefetch(ptr, numBytes);
    }
    sycl::event prefetch(const void* ptr, size_t numBytes, const std::vector<sycl::event>& deps) {
        for (auto e : deps) e.wait();
        return prefetch(ptr, numBytes);
    }
    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice, sycl::event dep) {
        dep.wait();
        return mem_advise(ptr, numBytes, advice);
    }
    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice, const std::vector<sycl::event>& deps) {
        for (auto e : deps) e.wait();
        return mem_advise(ptr, numBytes, advice);
    }

    ~cuda_threadpool() noexcept { release_stream_noexcept(); }

    bool stream_ordered_copy(void* dest, const void* src) const {
        if (!async_launch()) return false;
        return is_device_pointer(dest) && is_device_pointer(src);
    }

    static bool is_device_pointer(const void* p) {
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        return attr.type == cudaMemoryTypeDevice;
    }

private:
    sycl::device dev_;
    sycl::context ctx_{dev_};
    sycl::property_list props_;
    sycl::queue queue_;
    std::shared_ptr<int> id_{std::make_shared<int>()};
    std::shared_ptr<bool> stream_alive_;
    std::shared_ptr<void> stream_owner_;
    cudaDeviceProp device_props_{};
    bool device_props_valid_ = false;
    bool async_launch_ = [] {
        const char* env = std::getenv("PARAS_ASYNC_KERNELS");
        return env != nullptr && std::string(env) == "1";
    }();
    paras_shared_flag recording_;
    std::shared_ptr<paras_record_state> record_ = std::make_shared<paras_record_state>();

    [[noreturn]] void fail_recording(const char* what) const {
        record_->error = std::string(what) + " cannot be recorded into a graph";
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              record_->error);
    }

    static bool stream_recording(cudaStream_t s) {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        return cudaStreamIsCapturing(s, &status) == cudaSuccess && status == cudaStreamCaptureStatusActive;
    }

    void initialize_stream_for_device() {
        stream = nullptr;
        ownsStream = false;

        if (!dev_.is_gpu()) {
            return;
        }

        init_cuda_for(dev_);
        create_stream();
        ownsStream = true;
    }

    void ensure_stream() {
        if (stream == nullptr) {
            initialize_stream_for_device();
        }

        if (stream == nullptr) {
            throw std::runtime_error("CUDA operation requested without a valid CUDA stream");
        }

        if (dev_.is_gpu()) {
            const cudaError_t setErr = cudaSetDevice(dev_.get_native_id());
            if (setErr != cudaSuccess) {
                throw sycl::exception(cudaGetErrorString(setErr));
            }
        }

        if (!arch_checked_) {
            paras_detail::check_offload_arch_cuda(dev_.get_native_id());
            arch_checked_ = true;
        }
    }
    bool arch_checked_ = false;

    void create_stream() {
        cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

        if (err != cudaSuccess) {
            stream = nullptr;
            ownsStream = false;
            throw sycl::exception(cudaGetErrorString(err));
        }
        stream_alive_ = std::make_shared<bool>(true);
        const int device = dev_.is_gpu() ? dev_.get_native_id() : -1;
        std::shared_ptr<bool> alive = stream_alive_;
        stream_owner_ = std::shared_ptr<void>(static_cast<void*>(stream), [device, alive](void* p) {
            cudaStream_t s = static_cast<cudaStream_t>(p);
            if (device >= 0) {
                const cudaError_t setErr = cudaSetDevice(device);
                if (setErr != cudaSuccess)
                    std::cerr << "cudaSetDevice before cudaStreamDestroy failed: " << cudaGetErrorString(setErr) << '\n';
            }
            const cudaError_t syncErr = cudaStreamSynchronize(s);
            if (syncErr != cudaSuccess)
                std::cerr << "cudaStreamSynchronize before destroy failed: " << cudaGetErrorString(syncErr) << '\n';
            *alive = false;
            const cudaError_t err = cudaStreamDestroy(s);
            if (err != cudaSuccess) std::cerr << "cudaStreamDestroy failed: " << cudaGetErrorString(err) << '\n';
        });
    }

    void release_stream_noexcept() noexcept {
        stream_owner_.reset();
        stream = nullptr;
        ownsStream = false;
    }

    static void init_cuda_for(const sycl::device& d) {
        if (d.is_gpu()) {
            cudaError_t setErr = cudaSetDevice(d.get_native_id());
            if (setErr != cudaSuccess) {
                throw sycl::exception(cudaGetErrorString(setErr));
            }
        }

        cudaError_t err = cudaFree(nullptr);
        if (err != cudaSuccess) {
            throw sycl::exception(cudaGetErrorString(err));
        }
    }
};

inline cuda_threadpool& sycl::queue::get_or_create_gpu_pool() const {
    return gpu_pool_.get_or_create([this] { return std::make_shared<cuda_threadpool>(*this); });
}

inline sycl::event sycl::queue::paras_submit_event() const {
    cuda_threadpool& pool = get_or_create_gpu_pool();
    if (pool.async_launch() || !pool.paras_profiling()) {
        return pool.paras_finish_event(0);
    }
    return event{get_backend()};
}

#elif (PARAS_HIP_BACKEND)
#include <hip/hip_runtime.h>

class rocm_threadpool {

public:
    sycl::backend get_backend() const {
#if PARAS_HIP_BACKEND
        return sycl::backend::hip;
#else
        return sycl::backend::host;
#endif
    }

    sycl::device get_device() const { return dev_; }
    sycl::context get_context() const { return ctx_; }

    hipStream_t stream = nullptr;
    bool ownsStream = false;

    rocm_threadpool() : dev_(::paras_extension::select_device_no_selector()) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    explicit rocm_threadpool(const Selector& sel, const sycl::property_list& props = {})
        : dev_(::paras_extension::select_device_with_selector(sel)), props_(props) {
        initialize_stream_for_device();
    }

    template <typename Selector, typename = std::enable_if_t<std::is_invocable_r_v<
                                     int, const Selector&, const sycl::device&>>>
    rocm_threadpool(const Selector& sel, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : rocm_threadpool(sel, props) {}

    explicit rocm_threadpool(const sycl::property_list& props)
        : dev_(::paras_extension::select_device_no_selector()), props_(props) {
        initialize_stream_for_device();
    }

    explicit rocm_threadpool(const sycl::async_handler&, const sycl::property_list& props = {})
        : rocm_threadpool(props) {}

    rocm_threadpool(const sycl::device& dev, const sycl::async_handler&,
                    const sycl::property_list& props = {})
        : rocm_threadpool(dev, props) {}

    bool operator==(const rocm_threadpool& rhs) const noexcept { return id_ == rhs.id_; }
    bool operator!=(const rocm_threadpool& rhs) const noexcept { return !(*this == rhs); }

    rocm_threadpool(const sycl::queue& q) : dev_(q.get_device()), ctx_(q.get_context()), queue_(q) {
        queue_.paras_detach_pool();
        initialize_stream_for_device();
    }

    explicit rocm_threadpool(const sycl::context& ctx, const sycl::device& dev,
                             const sycl::property_list& props = {})
        : dev_(dev), ctx_(ctx), props_(props) {
        initialize_stream_for_device();
    }

    explicit rocm_threadpool(const sycl::device& dev, const sycl::property_list& props = {})
        : dev_(dev), props_(props) {
        initialize_stream_for_device();
    }

    rocm_threadpool(const rocm_threadpool& other)
        : stream(other.stream), ownsStream(other.ownsStream), dev_(other.dev_), ctx_(other.ctx_),
          props_(other.props_), queue_(other.queue_), id_(other.id_),
          stream_alive_(other.stream_alive_), stream_owner_(other.stream_owner_),
          recording_(other.recording_), record_(other.record_) {}

    rocm_threadpool& operator=(const rocm_threadpool& other) {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        dev_ = other.dev_;
        ctx_ = other.ctx_;
        props_ = other.props_;
        queue_ = other.queue_;
        id_ = other.id_;
        stream = other.stream;
        ownsStream = other.ownsStream;
        stream_alive_ = other.stream_alive_;
        stream_owner_ = other.stream_owner_;
        recording_ = other.recording_;
        record_ = other.record_;
        return *this;
    }

    rocm_threadpool(rocm_threadpool&& other) noexcept
        : stream(other.stream), ownsStream(other.ownsStream), dev_(std::move(other.dev_)),
          ctx_(std::move(other.ctx_)), props_(std::move(other.props_)),
          queue_(std::move(other.queue_)), id_(other.id_),
          stream_alive_(std::move(other.stream_alive_)), stream_owner_(std::move(other.stream_owner_)),
          recording_(other.recording_), record_(other.record_) {
        other.stream = nullptr;
        other.ownsStream = false;
    }

    rocm_threadpool& operator=(rocm_threadpool&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        release_stream_noexcept();

        stream = other.stream;
        ownsStream = other.ownsStream;
        stream_alive_ = std::move(other.stream_alive_);
        stream_owner_ = std::move(other.stream_owner_);
        recording_ = other.recording_;
        record_ = other.record_;
        dev_ = std::move(other.dev_);
        ctx_ = std::move(other.ctx_);
        props_ = std::move(other.props_);
        queue_ = std::move(other.queue_);
        id_ = other.id_;

        other.stream = nullptr;
        other.ownsStream = false;

        return *this;
    }

    std::uint64_t paras_profiling_start() const {
        return props_.has_profiling() ? sycl::paras_now_ns() : 0;
    }
    sycl::event paras_stream_event() const {
        hipStream_t s = stream;
        std::shared_ptr<bool> alive = stream_alive_;
        return sycl::event::paras_async(
            sycl::backend::hip,
            [s, alive] {
                if (!(alive && *alive)) return;
                if (stream_recording(s))
                    throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                                          "event::wait on a command recorded into a graph");
                const hipError_t err = hipStreamSynchronize(s);
                if (err != hipSuccess)
                    throw sycl::exception(std::string("kernel execution failed: ") +
                                          hipGetErrorString(err));
            },
            [s, alive] {
                return !(alive && *alive) ||
                       (!stream_recording(s) && hipStreamQuery(s) != hipErrorNotReady);
            });
    }
    sycl::event paras_finish_event(std::uint64_t t0) const {
        if (async_launch()) return paras_stream_event();
        if (!props_.has_profiling()) {
            return stream != nullptr ? paras_stream_event() : sycl::event{};
        }
        return sycl::event::paras_profiled(t0, t0, sycl::paras_now_ns());
    }

    bool paras_profiling() const noexcept { return props_.has_profiling(); }
    bool async_launch() const {
        return (async_launch_ || recording_) && !props_.has_profiling();
    }

    bool is_recording() const noexcept { return recording_; }

    void paras_record_host_task(std::function<void()> fn) {
        auto task = std::make_shared<std::function<void()>>(std::move(fn));
        const hipError_t err = hipLaunchHostFunc(
            stream, [](void* p) { (*static_cast<std::function<void()>*>(p))(); }, task.get());
        if (err != hipSuccess) fail_recording("a host task (hipLaunchHostFunc failed)");
        record_->host_tasks.push_back(std::move(task));
    }
    [[noreturn]] void paras_fail_recording(const char* what) const { fail_recording(what); }
    void paras_begin_recording();
    std::shared_ptr<void> paras_end_recording();
    std::shared_ptr<void> paras_instantiate_graph(void* graph);
    sycl::event paras_launch_graph(void* exec);
    sycl::ext::oneapi::experimental::queue_state ext_oneapi_get_state() const;
    sycl::event ext_oneapi_graph(const sycl::ext::oneapi::experimental::command_graph<
                                 sycl::ext::oneapi::experimental::graph_state::executable>& graph);

    const hipDeviceProp_t& device_props() {
        if (!device_props_valid_) {
            paras_rocm_detail_check_props(hipGetDeviceProperties(&device_props_, dev_.get_native_id()));
            device_props_valid_ = true;
        }
        return device_props_;
    }
    bool is_in_order() const { return props_.has_in_order(); }
    const void* paras_identity() const noexcept { return id_.get(); }
    template <typename Property>
    bool has_property() const noexcept {
        if constexpr (std::is_same_v<Property, sycl::property::queue::in_order>) {
            return props_.has_in_order();
        } else if constexpr (std::is_same_v<Property, sycl::property::queue::enable_profiling>) {
            return props_.has_profiling();
        } else {
            return false;
        }
    }

    template <typename Func>
    void submit(Func cgf) {
        sycl::handler cgh(*this);
        cgf(cgh);
    }

    static unsigned gpu_get_num_threads();

    template <typename Func>
    sycl::event spawn_1D(Func f);

    template <typename Func>
    sycl::event spawn_1D_event(Func f);

    template <typename Func>
    void spawn_ND(Func f);

    void wait() {
        if (recording_) fail_recording("queue::wait");
        ensure_stream();
        hipError_t errSync = hipStreamSynchronize(stream);
        if (errSync != hipSuccess) {
            throw sycl::exception(std::string("hipStreamSynchronize in queue::wait failed: ") +
                                  hipGetErrorString(errSync));
        }
    }

    void wait_and_throw() { wait(); }

    template <typename injectCustomFunc>
    sycl::event parasSYCL_enqueue_custom_operation(injectCustomFunc cgf) {
        sycl::handler cgh(*this);
        cgh.parasSYCL_enqueue_custom_operation(std::move(cgf));
        return paras_stream_event();
    }

    void wait() const {
        if (recording_) fail_recording("queue::wait");
        if (stream == nullptr) {
            throw sycl::exception("rocm_threadpool::wait called with null HIP stream");
        }

        hipError_t errSync = hipStreamSynchronize(stream);
        if (errSync != hipSuccess) {
            throw sycl::exception(std::string("hipStreamSynchronize in queue::wait failed: ") +
                                  hipGetErrorString(errSync));
        }
    }

    void wait_and_throw() const { wait(); }

    template <typename Func>
    void gpu_execute_1D(const sycl::range<1>& r, Func f);

    template <typename Func>
    void gpu_execute_1D_async(Func f);

    template <typename Func>
    void gpu_execute_2D(const sycl::range<2>& r, Func f);
    template <typename Func>
    void gpu_execute_3D(const sycl::range<3>& r, Func f);

    template <typename Func>
    void gpu_execute_nd_range_1D(const sycl::nd_range<1>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_2D(const sycl::nd_range<2>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename Func>
    void gpu_execute_nd_range_3D(const sycl::nd_range<3>& r, Func f,
                                 std::size_t sharedMemoryBytes = 0);

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(sycl::range<dim> r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

    template <typename KernelName = void, typename Func, int dim>
    sycl::event parallel_for(const sycl::nd_range<dim>& r, Func f) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        if constexpr (dim == 1) {
            gpu_execute_nd_range_1D(r, f);
        } else if constexpr (dim == 2) {
            gpu_execute_nd_range_2D(r, f);
        } else if constexpr (dim == 3) {
            gpu_execute_nd_range_3D(r, f);
        } else {
            static_assert(dim <= 3, "Only 1D/2D/3D supported");
        }
        return paras_finish_event(paras_t0);
    }

#include "kem/queue_shortcuts.hpp"

    sycl::event memcpy(void* dest, const void* src, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (recording_ && !stream_ordered_copy(dest, src))
            fail_recording("a copy to or from host memory");

        hipError_t err = hipMemcpyAsync(dest, src, numBytes, hipMemcpyDefault, stream);

        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemcpyAsync failed: ") +
                                     hipGetErrorString(err));
        }

        if (stream_ordered_copy(dest, src)) {
            return paras_finish_event(paras_t0);
        }

        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipStreamSynchronize after memcpy failed: ") +
                                     hipGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    template <typename T>
    sycl::event copy(const T* src, T* dest, size_t count) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (recording_ && !stream_ordered_copy(dest, src))
            fail_recording("a copy to or from host memory");

        hipError_t err = hipMemcpyAsync(static_cast<void*>(dest), static_cast<const void*>(src),
                                        count * sizeof(T), hipMemcpyDefault, stream);

        if (err != hipSuccess) {
            throw sycl::exception(hipGetErrorString(err));
        }

        if (stream_ordered_copy(dest, src)) {
            return paras_finish_event(paras_t0);
        }

        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw sycl::exception(std::string("hipStreamSynchronize after copy failed: ") +
                                  hipGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    sycl::event memset(void* ptr, int value, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();

        hipError_t err = hipMemsetAsync(ptr, value, numBytes, stream);

        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipMemsetAsync failed: ") +
                                     hipGetErrorString(err));
        }

        if (async_launch()) {
            return paras_finish_event(paras_t0);
        }

        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("hipStreamSynchronize after memset failed: ") +
                                     hipGetErrorString(err));
        }

        return paras_finish_event(paras_t0);
    }

    sycl::event prefetch(const void* ptr, size_t numBytes) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (numBytes != 0 && !recording_ &&
            hipMemPrefetchAsync(ptr, numBytes, dev_.get_native_id(), stream) != hipSuccess)
            (void)hipGetLastError();
        if (!async_launch()) (void)hipStreamSynchronize(stream);
        return paras_finish_event(paras_t0);
    }

    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice) {
        const std::uint64_t paras_t0 = paras_profiling_start();
        ensure_stream();
        if (numBytes != 0 && !recording_ &&
            hipMemAdvise(ptr, numBytes, static_cast<hipMemoryAdvise>(advice), dev_.get_native_id()) != hipSuccess)
            (void)hipGetLastError();
        return paras_finish_event(paras_t0);
    }

    sycl::event prefetch(const void* ptr, size_t numBytes, sycl::event dep) {
        dep.wait();
        return prefetch(ptr, numBytes);
    }
    sycl::event prefetch(const void* ptr, size_t numBytes, const std::vector<sycl::event>& deps) {
        for (auto e : deps) e.wait();
        return prefetch(ptr, numBytes);
    }
    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice, sycl::event dep) {
        dep.wait();
        return mem_advise(ptr, numBytes, advice);
    }
    sycl::event mem_advise(const void* ptr, size_t numBytes, int advice, const std::vector<sycl::event>& deps) {
        for (auto e : deps) e.wait();
        return mem_advise(ptr, numBytes, advice);
    }

    ~rocm_threadpool() noexcept { release_stream_noexcept(); }

    bool stream_ordered_copy(void* dest, const void* src) const {
        if (!async_launch()) return false;
        return is_device_pointer(dest) && is_device_pointer(src);
    }

    static bool is_device_pointer(const void* p) {
        hipPointerAttribute_t attr{};
        if (hipPointerGetAttributes(&attr, p) != hipSuccess) {
            (void)hipGetLastError();
            return false;
        }
        return attr.type == hipMemoryTypeDevice;
    }

private:
    sycl::device dev_;
    sycl::context ctx_{dev_};
    sycl::property_list props_;
    sycl::queue queue_;
    std::shared_ptr<int> id_{std::make_shared<int>()};
    std::shared_ptr<bool> stream_alive_;
    std::shared_ptr<void> stream_owner_;
    hipDeviceProp_t device_props_{};
    bool device_props_valid_ = false;
    static void paras_rocm_detail_check_props(hipError_t err) {
        if (err != hipSuccess)
            throw sycl::exception(std::string("hipGetDeviceProperties failed: ") +
                                  hipGetErrorString(err));
    }
    bool async_launch_ = [] {
        const char* env = std::getenv("PARAS_ASYNC_KERNELS");
        return env != nullptr && std::string(env) == "1";
    }();
    paras_shared_flag recording_;
    std::shared_ptr<paras_record_state> record_ = std::make_shared<paras_record_state>();

    [[noreturn]] void fail_recording(const char* what) const {
        record_->error = std::string(what) + " cannot be recorded into a graph";
        throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                              record_->error);
    }

    static bool stream_recording(hipStream_t s) {
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        return hipStreamIsCapturing(s, &status) == hipSuccess && status == hipStreamCaptureStatusActive;
    }

    void initialize_stream_for_device() {
        stream = nullptr;
        ownsStream = false;

        if (!dev_.is_gpu()) {
            return;
        }

        init_rocm_for(dev_);
        create_stream();
        ownsStream = true;
    }

    void ensure_stream() {
        if (stream == nullptr) {
            initialize_stream_for_device();
        }

        if (stream == nullptr) {
            throw std::runtime_error("HIP operation requested without a valid HIP stream");
        }

        if (dev_.is_gpu()) {
            const hipError_t setErr = hipSetDevice(dev_.get_native_id());
            if (setErr != hipSuccess) {
                throw sycl::exception(hipGetErrorString(setErr));
            }
        }

        if (!arch_checked_) {
            paras_detail::check_offload_arch_hip(dev_.get_native_id());
            arch_checked_ = true;
        }
    }
    bool arch_checked_ = false;

    void create_stream() {
        hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

        if (err != hipSuccess) {
            stream = nullptr;
            ownsStream = false;
            throw sycl::exception(hipGetErrorString(err));
        }
        stream_alive_ = std::make_shared<bool>(true);
        const int device = dev_.is_gpu() ? dev_.get_native_id() : -1;
        std::shared_ptr<bool> alive = stream_alive_;
        stream_owner_ = std::shared_ptr<void>(static_cast<void*>(stream), [device, alive](void* p) {
            hipStream_t s = static_cast<hipStream_t>(p);
            if (device >= 0) {
                const hipError_t setErr = hipSetDevice(device);
                if (setErr != hipSuccess)
                    std::cerr << "hipSetDevice before hipStreamDestroy failed: " << hipGetErrorString(setErr) << '\n';
            }
            const hipError_t syncErr = hipStreamSynchronize(s);
            if (syncErr != hipSuccess)
                std::cerr << "hipStreamSynchronize before destroy failed: " << hipGetErrorString(syncErr) << '\n';
            *alive = false;
            const hipError_t err = hipStreamDestroy(s);
            if (err != hipSuccess) std::cerr << "hipStreamDestroy failed: " << hipGetErrorString(err) << '\n';
        });
    }

    void release_stream_noexcept() noexcept {
        stream_owner_.reset();
        stream = nullptr;
        ownsStream = false;
    }

    static void init_rocm_for(const sycl::device& d) {
        if (d.is_gpu()) {
            hipError_t setErr = hipSetDevice(d.get_native_id());
            if (setErr != hipSuccess) {
                throw sycl::exception(hipGetErrorString(setErr));
            }
        }

        hipError_t err = hipFree(nullptr);
        if (err != hipSuccess) {
            throw sycl::exception(hipGetErrorString(err));
        }
    }
};

inline rocm_threadpool& sycl::queue::get_or_create_rocm_pool() const {
    return rocm_pool_.get_or_create([this] { return std::make_shared<rocm_threadpool>(*this); });
}

inline sycl::event sycl::queue::paras_submit_event() const {
    rocm_threadpool& pool = get_or_create_rocm_pool();
    if (pool.async_launch() || !pool.paras_profiling()) {
        return pool.paras_finish_event(0);
    }
    return event{get_backend()};
}

#endif

#include "kem_gpu/gpu_threadpool_execute_1D.hpp"
#include "kem_gpu/gpu_threadpool_execute_1D_async.hpp"
#include "kem_gpu/gpu_threadpool_execute_ND.hpp"
#include "kem_gpu/gpu_threadpool_execute_common.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_1D.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_2D.hpp"
#include "kem_gpu/gpu_threadpool_execute_nd_range_3D.hpp"

namespace sycl {

inline void handler::wait_for_prior_commands() {
    wait_for_dependencies();
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_ != nullptr && gpu_pool_->get_device().is_gpu()) gpu_pool_->wait();
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ != nullptr && rocm_pool_->get_device().is_gpu()) rocm_pool_->wait();
#endif
}

inline int handler::paras_activate_native_device() {
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_ == nullptr || !gpu_pool_->get_device().is_gpu()) return -1;
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
    const cudaError_t err = cudaSetDevice(gpu_pool_->get_device().get_native_id());
    if (err != cudaSuccess) throw sycl::exception(cudaGetErrorString(err));
    return previous;
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ == nullptr || !rocm_pool_->get_device().is_gpu()) return -1;
    int previous = -1;
    if (hipGetDevice(&previous) != hipSuccess) previous = -1;
    const hipError_t err = hipSetDevice(rocm_pool_->get_device().get_native_id());
    if (err != hipSuccess) throw sycl::exception(hipGetErrorString(err));
    return previous;
#else
    return -1;
#endif
}

inline void handler::paras_restore_native_device(int previous) noexcept {
    if (previous < 0) return;
#if (PARAS_CUDA_BACKEND)
    (void)cudaSetDevice(previous);
#elif (PARAS_HIP_BACKEND)
    (void)hipSetDevice(previous);
#endif
}

inline void handler::memset(void* ptr, int value, size_t num_bytes) {
#if PARAS_CUDA_BACKEND
    if (gpu_pool_ == nullptr) {
        throw std::runtime_error("handler::memset has no CUDA backend");
    }
    wait_for_dependencies();
    (void)gpu_pool_->memset(ptr, value, num_bytes);
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ == nullptr) {
        throw std::runtime_error("handler::memset has no HIP backend");
    }
    wait_for_dependencies();
    (void)rocm_pool_->memset(ptr, value, num_bytes);
#endif
}

inline void handler::memcpy(void* dest, const void* src, size_t num_bytes) {
#if PARAS_CUDA_BACKEND
    if (gpu_pool_ == nullptr) {
        throw std::runtime_error("handler::memcpy has no CUDA backend");
    }
    wait_for_dependencies();
    (void)gpu_pool_->memcpy(dest, src, num_bytes);
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_ == nullptr) {
        throw std::runtime_error("handler::memcpy has no HIP backend");
    }
    wait_for_dependencies();
    (void)rocm_pool_->memcpy(dest, src, num_bytes);
#endif
}

#if (PARAS_CUDA_BACKEND)
#define PARAS_HANDLER_POOL gpu_pool_
#elif (PARAS_HIP_BACKEND)
#define PARAS_HANDLER_POOL rocm_pool_
#endif

inline bool handler::paras_recording() const {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    return PARAS_HANDLER_POOL != nullptr && PARAS_HANDLER_POOL->is_recording();
#else
    return false;
#endif
}

inline void handler::paras_record_host_task(std::function<void()> fn) {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    wait_for_dependencies();
    PARAS_HANDLER_POOL->paras_record_host_task(std::move(fn));
#else
    fn();
#endif
}

inline void handler::paras_fail_recording(const char* what) const {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    if (PARAS_HANDLER_POOL != nullptr) PARAS_HANDLER_POOL->paras_fail_recording(what);
#endif
    throw sycl::exception(sycl::make_error_code(sycl::errc::feature_not_supported),
                          std::string(what) + " cannot be recorded into a graph");
}

inline void handler::paras_prefetch(const void* ptr, std::size_t num_bytes) {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    if (PARAS_HANDLER_POOL == nullptr || !PARAS_HANDLER_POOL->get_device().is_gpu()) return;
    wait_for_dependencies();
    (void)PARAS_HANDLER_POOL->prefetch(ptr, num_bytes);
#else
    (void)ptr;
    (void)num_bytes;
#endif
}

inline void handler::paras_mem_advise(const void* ptr, std::size_t num_bytes, int advice) {
#if (PARAS_CUDA_BACKEND || PARAS_HIP_BACKEND)
    if (PARAS_HANDLER_POOL == nullptr || !PARAS_HANDLER_POOL->get_device().is_gpu()) return;
    wait_for_dependencies();
    (void)PARAS_HANDLER_POOL->mem_advise(ptr, num_bytes, advice);
#else
    (void)ptr;
    (void)num_bytes;
    (void)advice;
#endif
}
#undef PARAS_HANDLER_POOL

inline void* sycl::interop_handle::get_native_queue() {
    return backend_ptr_;
}

template <>
inline sycl::backend_return_t<sycl::backend::cuda, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::cuda>() const {
    if (backend_ != sycl::backend::cuda) {
        return nullptr;
    }
#if (PARAS_CUDA_BACKEND)
    auto* pool = static_cast<cuda_threadpool*>(backend_ptr_);
    return pool ? pool->stream : nullptr;
#elif (PARAS_HIP_BACKEND)
    return static_cast<sycl::backend_return_t<sycl::backend::cuda, sycl::queue>>(backend_ptr_);
#endif
}

template <>
inline sycl::backend_return_t<sycl::backend::host, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::host>() const {
    if (backend_ != sycl::backend::host) {
        return nullptr;
    }

    return backend_ptr_;
}

template <>
inline sycl::backend_return_t<sycl::backend::hip, sycl::queue>
sycl::interop_handle::get_native_queue<sycl::backend::hip>() const {
    if (backend_ != sycl::backend::hip) {
        return nullptr;
    }
#if (PARAS_CUDA_BACKEND)
    return backend_ptr_;
#elif (PARAS_HIP_BACKEND)
    auto* pool = static_cast<rocm_threadpool*>(backend_ptr_);
    return pool ? pool->stream : nullptr;
#endif
}
} // namespace sycl

#if (PARAS_CUDA_BACKEND)
template <typename Func>
sycl::event cuda_threadpool::spawn_1D(Func f) {
    const std::uint64_t paras_t0 = paras_profiling_start();

    sycl::handler cgh(*this);
    f(cgh);

    return paras_finish_event(paras_t0);
}

template <typename Func>
sycl::event cuda_threadpool::spawn_1D_event(Func f) {
    return spawn_1D(f);
}

template <typename Func>
void cuda_threadpool::spawn_ND(Func f) {
    sycl::handler cgh(*this); // Currently same as 1D; modify for ND ranges
    f(cgh);
}
#elif (PARAS_HIP_BACKEND)
template <typename Func>
sycl::event rocm_threadpool::spawn_1D(Func f) {
    const std::uint64_t paras_t0 = paras_profiling_start();

    sycl::handler cgh(*this);
    f(cgh);

    return paras_finish_event(paras_t0);
}

template <typename Func>
sycl::event rocm_threadpool::spawn_1D_event(Func f) {
    return spawn_1D(f);
}

template <typename Func>
void rocm_threadpool::spawn_ND(Func f) {
    sycl::handler cgh(*this); // Currently same as 1D; modify for ND ranges
    f(cgh);
}
#endif

#include "kem_gpu/handler_impl.hpp"
#include "kem_gpu/command_graph_impl.hpp"

inline void sycl::queue::wait() const {
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_) {
        gpu_pool_->wait();
    }
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_) {
        rocm_pool_->wait();
    }
#endif
}

inline void sycl::queue::wait_and_throw() const {
#if (PARAS_CUDA_BACKEND)
    if (gpu_pool_) {
        gpu_pool_->wait_and_throw();
    }

    throw_async_exceptns();
#elif (PARAS_HIP_BACKEND)
    if (rocm_pool_) {
        rocm_pool_->wait_and_throw();
    }

    throw_async_exceptns();
#endif
}

#endif
