"""Optional native host launch of the unchanged Triton Sage quantizer.

The extension compiles host C++ only. Plans retain their Triton kernels, never
input/output tensors, and reject an unsupported driver argument ABI. A missing
compiler or incompatible runtime leaves the regular Triton path available.
"""

from functools import lru_cache
import re
import warnings

import torch


_SOURCE = r"""
#include <array>
#include <algorithm>
#include <tuple>
#include <cuda.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

class SageQuantPlan {
public:
    explicit SageQuantPlan(const std::vector<int64_t>& spec) {
        TORCH_CHECK(spec.size() == 21, "Invalid Sage quantization launch plan");
        std::copy(spec.begin(), spec.end(), spec_.begin());
        c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(spec_[17]));
        const int expected[] = {4, 4, 10};
        for (int kernel = 0; kernel < 3; ++kernel) {
            if (kernel == 1 && spec_[9]) {
                continue;
            }
            auto function = reinterpret_cast<CUfunction>(spec_[kernel]);
            TORCH_CHECK(function && spec_[3 + kernel] > 0,
                        "Missing Sage quantization kernel");
            // Triton may append zero-sized global and profiling scratch pointers.
            // The exact count comes from the compiled PTX entry. Query only
            // valid indices so sanitizers never see an intentional API error.
            int count = spec_[18 + kernel];
            TORCH_CHECK(count >= expected[kernel] && count <= expected[kernel] + 2,
                        "Unsupported Triton scratch parameter ABI");
            for (int param = 0; param < count; ++param) {
                size_t offset = 0, size = 0;
                check_cuda(cuFuncGetParamInfo(function, param, &offset, &size));
                TORCH_CHECK(size == sizeof(void*), "Unsupported Triton parameter ABI");
            }
        }
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor,
               at::Tensor, at::Tensor, at::Tensor>
    run(const at::Tensor& mQ, const at::Tensor& mK, const at::Tensor& mV) const {
        namespace py = pybind11;
        if (mQ.dim() != 4 || mK.dim() != 4 || mV.dim() != 4) {
            throw py::value_error("Q, K, and V must be rank-4 BHSD tensors");
        }
        if (mQ.scalar_type() != at::kBFloat16 || mK.scalar_type() != at::kBFloat16 ||
            mV.scalar_type() != at::kBFloat16) {
            throw py::type_error("Sage FP8 quantization currently requires BF16 inputs");
        }
        if (!mQ.is_cuda() || !mK.is_cuda() || !mV.is_cuda()) {
            throw py::value_error("Q, K, and V must be CUDA tensors");
        }
        if (mQ.device() != mK.device() || mQ.device() != mV.device()) {
            throw py::value_error("Q, K, and V must be on the same CUDA device");
        }
        if (!mQ.is_contiguous() || !mK.is_contiguous() || !mV.is_contiguous()) {
            throw py::value_error("Q, K, and V must use contiguous BHSD storage");
        }
        int64_t batch = spec_[13], heads = spec_[14], sq = spec_[15], sk = spec_[16];
        if (mQ.sizes() != at::IntArrayRef({batch, heads, sq, 128}) ||
            mK.sizes() != at::IntArrayRef({batch, heads, sk, 128}) ||
            mV.sizes() != mK.sizes() || mQ.get_device() != spec_[17]) {
            throw py::value_error("Q/K/V shape or device does not match quantization plan");
        }
        c10::cuda::CUDAGuard guard(mQ.device());
        auto stream = reinterpret_cast<CUstream>(
            c10::cuda::getCurrentCUDAStream(mQ.get_device()).stream());
        auto fp8 = mQ.options().dtype(at::ScalarType::Float8_e4m3fn);
        auto fp32 = mQ.options().dtype(at::kFloat);
        auto mQ8 = at::empty(mQ.sizes(), fp8);
        auto mK8 = at::empty(mK.sizes(), fp8);
        auto mV8 = at::empty(mV.sizes(), fp8);
        auto mQScale = at::empty({batch, heads, sq}, fp32);
        auto mKScale = at::empty({batch, heads, (sk + 15) / 16}, fp32);
        auto mVScale = at::empty({heads, 128}, fp32);
        auto mKMean = at::empty({batch, heads, 128}, fp32);
        auto mPartial = spec_[9] ? at::Tensor() :
            at::empty({2, batch, heads, spec_[11], 128}, fp32);

        void* scratch = nullptr;
        void *q = mQ.data_ptr(), *k = mK.data_ptr(), *v = mV.data_ptr();
        void *q8 = mQ8.data_ptr(), *k8 = mK8.data_ptr(), *v8 = mV8.data_ptr();
        void *qs = mQScale.data_ptr(), *ks = mKScale.data_ptr();
        void *vs = mVScale.data_ptr(), *km = mKMean.data_ptr();
        void* partial_sum = spec_[9] ? nullptr : mPartial.data_ptr();
        void* partial_max = spec_[9] ? nullptr :
            static_cast<void*>(static_cast<char*>(partial_sum) + spec_[10] * sizeof(float));
        void *stats_k = spec_[9] ? km : partial_sum;
        void *stats_v = spec_[9] ? vs : partial_max;
        void* stats_args[] = {&k, &v, &stats_k, &stats_v, &scratch, &scratch};
        launch(0, spec_[11], 4, batch * heads, stream, stats_args);
        if (!spec_[9]) {
            void* finish_args[] = {&partial_sum, &partial_max, &km, &vs, &scratch, &scratch};
            launch(1, heads, 128 / spec_[12], 1, stream, finish_args);
        }
        void* quant_args[] = {&q, &k, &v, &q8, &k8, &v8, &qs, &ks, &vs, &km,
                             &scratch, &scratch};
        launch(2, (std::max(sq, sk) + 31) / 32, batch * heads, 1, stream, quant_args);
        return {mQ8, mK8, mV8, mQScale, mKScale, mVScale};
    }

private:
    std::array<int64_t, 21> spec_;

    static void check_cuda(CUresult result) {
        if (result != CUDA_SUCCESS) {
            const char* message = nullptr;
            cuGetErrorString(result, &message);
            TORCH_CHECK(false, "Sage quantization launch: ",
                        message ? message : "unknown CUDA driver error");
        }
    }

    void launch(int kernel, unsigned x, unsigned y, unsigned z,
                CUstream stream, void** args) const {
        check_cuda(cuLaunchKernel(reinterpret_cast<CUfunction>(spec_[kernel]), x, y, z,
                                 32 * spec_[3 + kernel], 1, 1, spec_[6 + kernel],
                                 stream, args, nullptr));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    pybind11::class_<SageQuantPlan>(module, "SageQuantPlan")
        .def(pybind11::init<const std::vector<int64_t>&>())
        .def("run", &SageQuantPlan::run);
}
"""


@lru_cache(maxsize=1)
def _runtime():
    # Never build a host extension while CUDA graph capture is active.
    if torch.cuda.is_current_stream_capturing():
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        return load_inline(
            "bsa_sage_quant_host",
            cpp_sources=_SOURCE,
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
        )
    except (ImportError, OSError, RuntimeError) as error:
        warnings.warn(
            f"Native Sage quantization host runtime unavailable; using Triton: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def make_quant_plan(kernels, shape, device_index):
    """Return a checked launcher plus kernel lifetime owners, or None."""
    module = _runtime()
    if module is None:
        return None
    batch, heads, sq, sk = shape
    splits = min((sk + 127) // 128, 128)
    cols = max(1, 32 // (1 << (batch - 1).bit_length()))
    parameter_counts = []
    for kernel, expected in zip(kernels, (4, 4, 10)):
        if kernel is not None:
            metadata = kernel.metadata
            if (getattr(metadata, "global_scratch_size", 0)
                    or getattr(metadata, "profile_scratch_size", 0)
                    or metadata.num_ctas != 1):
                return None
            entry = re.search(r"\.entry\s+" + re.escape(kernel.name) + r"\s*\((.*?)\)",
                              kernel.asm.get("ptx", ""), re.DOTALL)
            if entry is None:
                return None
            parameters = entry.group(1).split(",")
            if (not expected <= len(parameters) <= expected + 2
                    or any(re.fullmatch(
                        r"\s*\.param\s+\.u64(?:\s+\.ptr(?:\s+\.global)?"
                        r"(?:\s+\.align\s+\d+)?)?\s+[\w$]+\s*", p) is None
                           for p in parameters)):
                return None
            parameter_counts.append(len(parameters))
        else:
            parameter_counts.append(0)
    spec = (
        [int(kernel.function) if kernel else 0 for kernel in kernels]
        + [kernel.metadata.num_warps if kernel else 0 for kernel in kernels]
        + [kernel.metadata.shared if kernel else 0 for kernel in kernels]
        + [int(batch == 1 and splits == 1), batch * heads * splits * 128,
           splits, cols, batch, heads, sq, sk, device_index]
        + parameter_counts
    )
    try:
        plan = module.SageQuantPlan(spec)
    except RuntimeError as error:
        warnings.warn(f"Unsupported native quantization plan; using Triton: {error}",
                      RuntimeWarning, stacklevel=2)
        return None
    return plan.run, tuple(kernels)
