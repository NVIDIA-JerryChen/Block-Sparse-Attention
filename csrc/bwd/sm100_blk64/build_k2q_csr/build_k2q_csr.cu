// Runtime-topK CUDA q2k -> k2q CSR builder for BSA blk64 backward.
//
// The pipeline follows the MiniMax CSR builder structure but is specialized
// for BSA's non-varlen block layout:
//   q2k:      [B, H, Q_blocks, max_kv]
//   row_ptr:  [B, H, KV_blocks + 1]
//   q_idx:    [total_edges]
//
// It intentionally does not specialize on fixed topK values. Fixed-count and
// q2k_block_nums paths share the same runtime-topK kernels.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdlib>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == at::kInt, #x " must be int32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_INT(x)

namespace {

constexpr int kWarpSize = 32;
constexpr int kScheduleModeNone = 0;
constexpr int kScheduleModeRowChunk = 1;
constexpr int kScheduleModeQRange = 2;

__device__ __forceinline__ int atomic_inc_int16_packed(int* base_int32, int row) {
    int idx = row >> 1;
    int shift = (row & 1) << 4;
    int delta = 1 << shift;
    int old = atomicAdd(&base_int32[idx], delta);
    return (old >> shift) & 0xFFFF;
}

__device__ __forceinline__ int read_int16_packed(int const* base_int32, int row) {
    int v = base_int32[row >> 1];
    int shift = (row & 1) << 4;
    return (v >> shift) & 0xFFFF;
}

__device__ __forceinline__ int valid_count_for_q(
    int const* __restrict__ q2k_nums,
    bool has_variable_nums,
    int block_sparse_num,
    int B,
    int H,
    int Q,
    int b,
    int h,
    int q)
{
    if (!has_variable_nums) {
        return block_sparse_num;
    }
    return q2k_nums[((b * H + h) * Q) + q];
}

template <int kWarps>
__global__ void k2q_hist_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ q2k_nums,
    int* __restrict__ row_counts,
    int* __restrict__ tile_counts,
    int B,
    int H,
    int Q,
    int max_kv,
    int num_kv_blocks,
    int block_sparse_num,
    bool has_variable_nums,
    int q_per_cta,
    int q_per_warp)
{
    constexpr int kThreads = kWarps * kWarpSize;
    extern __shared__ int smem_hist_int[];

    int g = blockIdx.x;
    int b = blockIdx.y;
    int h = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int q_start_cta = g * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    int packed_per_warp = (num_kv_blocks + 1) >> 1;
    int* my_hist = smem_hist_int + warp_id * packed_per_warp;

    for (int i = lane; i < packed_per_warp; i += kWarpSize) {
        my_hist[i] = 0;
    }
    __syncthreads();

    for (int qi = q_start_warp; qi < q_end_warp; ++qi) {
        int n = valid_count_for_q(
            q2k_nums, has_variable_nums, block_sparse_num, B, H, Q, b, h, qi);
        n = min(n, max_kv);
        int const* q_base = q2k + (((b * H + h) * Q + qi) * max_kv);
        for (int slot_base = 0; slot_base < n; slot_base += kWarpSize) {
            int slot = slot_base + lane;
            if (slot < n) {
                int kv = q_base[slot];
                if (kv >= 0 && kv < num_kv_blocks) {
                    atomic_inc_int16_packed(my_hist, kv);
                }
            }
        }
    }
    __syncthreads();

    int bh = b * H + h;
    int g_total_idx = g * kWarps + warp_id;
    int* my_tile = tile_counts + ((g_total_idx * B * H + bh) * num_kv_blocks);
    for (int kv = lane; kv < num_kv_blocks; kv += kWarpSize) {
        my_tile[kv] = read_int16_packed(my_hist, kv);
    }
    __syncthreads();

    int* my_counts = row_counts + bh * num_kv_blocks;
    for (int kv = tid; kv < num_kv_blocks; kv += kThreads) {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            sum += read_int16_packed(smem_hist_int + w * packed_per_warp, kv);
        }
        if (sum != 0) {
            atomicAdd(my_counts + kv, sum);
        }
    }
}

template <int kThreads>
__global__ void k2q_row_prefix_kernel(
    int const* __restrict__ row_counts,
    int const* __restrict__ bh_offsets,
    int* __restrict__ row_ptr,
    int* __restrict__ schedule_metadata,
    int* __restrict__ schedule_work_counts,
    int B,
    int H,
    int num_kv_blocks,
    int fixed_edges_per_bh,
    bool has_variable_nums,
    int schedule_capacity_per_bh,
    int target_q_per_cta)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;
    int bh = b * H + h;

    __shared__ int scan_buf[kThreads];
    int const* counts = row_counts + bh * num_kv_blocks;
    int* ptr = row_ptr + bh * (num_kv_blocks + 1);
    int base_offset = has_variable_nums ? bh_offsets[bh] : bh * fixed_edges_per_bh;

    int chunk = (num_kv_blocks + kThreads - 1) / kThreads;
    int lo = tid * chunk;
    int hi = min(lo + chunk, num_kv_blocks);

    int local_sum = 0;
    for (int i = lo; i < hi; ++i) {
        local_sum += counts[i];
    }
    scan_buf[tid] = local_sum;
    __syncthreads();

    for (int off = 1; off < kThreads; off <<= 1) {
        int add = (tid >= off) ? scan_buf[tid - off] : 0;
        __syncthreads();
        scan_buf[tid] += add;
        __syncthreads();
    }

    if (tid == 0) {
        ptr[0] = base_offset;
    }

    int running = base_offset + scan_buf[tid] - local_sum;
    for (int i = lo; i < hi; ++i) {
        int row_count = counts[i];
        running += row_count;
        ptr[i + 1] = running;
        if (schedule_metadata != nullptr && row_count > 0) {
            int row_start = running - row_count;
            int num_chunks = (row_count + target_q_per_cta - 1) / target_q_per_cta;
            int base = atomicAdd(schedule_work_counts + bh, num_chunks);
            for (int c = 0; c < num_chunks; ++c) {
                int work_idx = base + c;
                if (work_idx < schedule_capacity_per_bh) {
                    int q_begin = c * target_q_per_cta;
                    int q_count = min(target_q_per_cta, row_count - q_begin);
                    int* meta = schedule_metadata +
                        (((size_t)bh * schedule_capacity_per_bh + work_idx) * 4);
                    meta[0] = i;
                    meta[1] = row_start + q_begin;
                    meta[2] = q_count;
                    meta[3] = 0;
                }
            }
        }
    }
}

template <int kThreads, int kRowsPerBlock>
__global__ void k2q_tile_prefix_smem_kernel(
    int* __restrict__ tile_counts,
    int const* __restrict__ row_ptr,
    int B,
    int H,
    int num_kv_blocks,
    int G_total)
{
    extern __shared__ int smem_tprefix[];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int blocks_per_bh = (num_kv_blocks + kRowsPerBlock - 1) / kRowsPerBlock;
    int job = blockIdx.x;
    int bh = job / blocks_per_bh;
    int block_in_bh = job - bh * blocks_per_bh;
    if (bh >= B * H) return;

    int base_kv = block_in_bh * kRowsPerBlock;
    if (base_kv >= num_kv_blocks) return;
    int actual_rows = min(kRowsPerBlock, num_kv_blocks - base_kv);

    size_t stride_g = (size_t)B * H * num_kv_blocks;
    int* base_ptr = tile_counts + (size_t)bh * num_kv_blocks + base_kv;
    int total_elems = G_total * actual_rows;

    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_rows;
        int g = i / actual_rows;
        smem_tprefix[r_off * G_total + g] = base_ptr[g * stride_g + r_off];
    }
    __syncthreads();

    if (warp_id < actual_rows) {
        int kv = base_kv + warp_id;
        int running = row_ptr[(size_t)bh * (num_kv_blocks + 1) + kv];
        int* my_row = smem_tprefix + warp_id * G_total;
        for (int g0 = 0; g0 < G_total; g0 += kWarpSize) {
            int g = g0 + lane;
            int v = (g < G_total) ? my_row[g] : 0;
            int x = v;
            #pragma unroll
            for (int off = 1; off < kWarpSize; off <<= 1) {
                int nbr = __shfl_up_sync(0xFFFFFFFF, x, off);
                if (lane >= off) x += nbr;
            }
            int excl = running + x - v;
            if (g < G_total) {
                my_row[g] = excl;
            }
            int chunk_sum = __shfl_sync(0xFFFFFFFF, x, 31);
            running += chunk_sum;
        }
    }
    __syncthreads();

    for (int i = tid; i < total_elems; i += kThreads) {
        int r_off = i % actual_rows;
        int g = i / actual_rows;
        base_ptr[g * stride_g + r_off] = smem_tprefix[r_off * G_total + g];
    }
}

template <int kWarps>
__global__ void k2q_scatter_kernel(
    int const* __restrict__ q2k,
    int const* __restrict__ q2k_nums,
    int const* __restrict__ tile_offsets,
    int* __restrict__ q_idx,
    int B,
    int H,
    int Q,
    int max_kv,
    int num_kv_blocks,
    int q_idx_capacity,
    int block_sparse_num,
    bool has_variable_nums,
    int q_per_cta,
    int q_per_warp)
{
    extern __shared__ int smem_cursor_int[];

    int g = blockIdx.x;
    int b = blockIdx.y;
    int h = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int q_start_cta = g * q_per_cta;
    int q_end_cta = min(q_start_cta + q_per_cta, Q);
    int q_start_warp = min(q_start_cta + warp_id * q_per_warp, q_end_cta);
    int q_end_warp = min(q_start_warp + q_per_warp, q_end_cta);

    int packed_per_warp = (num_kv_blocks + 1) >> 1;
    int* my_cursor = smem_cursor_int + warp_id * packed_per_warp;

    for (int i = lane; i < packed_per_warp; i += kWarpSize) {
        my_cursor[i] = 0;
    }
    __syncthreads();

    int bh = b * H + h;
    int g_total_idx = g * kWarps + warp_id;
    int const* my_offsets = tile_offsets + ((g_total_idx * B * H + bh) * num_kv_blocks);

    for (int qi = q_start_warp; qi < q_end_warp; ++qi) {
        int n = valid_count_for_q(
            q2k_nums, has_variable_nums, block_sparse_num, B, H, Q, b, h, qi);
        n = min(n, max_kv);
        int const* q_base = q2k + (((b * H + h) * Q + qi) * max_kv);
        for (int slot_base = 0; slot_base < n; slot_base += kWarpSize) {
            int slot = slot_base + lane;
            if (slot < n) {
                int kv = q_base[slot];
                if (kv >= 0 && kv < num_kv_blocks) {
                    int local_slot = atomic_inc_int16_packed(my_cursor, kv);
                    int out_pos = my_offsets[kv] + local_slot;
                    if (out_pos >= 0 && out_pos < q_idx_capacity) {
                        q_idx[out_pos] = qi;
                    }
                }
            }
        }
    }
}

__global__ void k2q_qrange_schedule_kernel(
    int const* __restrict__ row_ptr,
    int const* __restrict__ q_idx,
    int* __restrict__ schedule_metadata,
    int* __restrict__ schedule_work_counts,
    int B,
    int H,
    int num_kv_blocks,
    int num_q_groups,
    int q_bucket_size_blocks,
    int schedule_capacity_per_bh)
{
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (work_idx >= schedule_capacity_per_bh) {
        return;
    }

    int bh = b * H + h;
    if (work_idx == 0) {
        schedule_work_counts[bh] = schedule_capacity_per_bh;
    }

    int q_group = work_idx / num_kv_blocks;
    int kv = work_idx - q_group * num_kv_blocks;
    int* meta = schedule_metadata + (((size_t)bh * schedule_capacity_per_bh + work_idx) * 4);
    if (q_group >= num_q_groups || kv >= num_kv_blocks) {
        meta[0] = 0;
        meta[1] = 0;
        meta[2] = 0;
        meta[3] = q_group;
        return;
    }

    int row_start = row_ptr[(size_t)bh * (num_kv_blocks + 1) + kv];
    int row_end = row_ptr[(size_t)bh * (num_kv_blocks + 1) + kv + 1];
    int q_lo = q_group * q_bucket_size_blocks;
    int q_hi = q_lo + q_bucket_size_blocks;

    int lo = row_start;
    int hi = row_end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int q = q_idx[mid];
        if (q < q_lo) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int slice_start = lo;

    hi = row_end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int q = q_idx[mid];
        if (q < q_hi) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    meta[0] = kv;
    meta[1] = slice_start;
    meta[2] = lo - slice_start;
    meta[3] = q_group;
}

template <int kWarps>
void launch_hist_scatter(
    torch::Tensor q2k,
    torch::Tensor q2k_nums,
    torch::Tensor bh_offsets,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor schedule_metadata,
    torch::Tensor schedule_work_counts,
    torch::Tensor row_counts,
    torch::Tensor tile_counts,
    int B,
    int H,
    int Q,
    int max_kv,
    int num_kv_blocks,
    int q_idx_capacity,
    int fixed_edges_per_bh,
    int block_sparse_num,
    bool has_variable_nums,
    bool emit_schedule,
    int schedule_mode,
    int schedule_capacity_per_bh,
    int target_q_per_cta,
    int qrange_num_q_groups,
    int qrange_q_bucket_size_blocks,
    int G,
    int q_per_cta,
    int q_per_warp,
    cudaStream_t stream)
{
    int packed_per_warp = (num_kv_blocks + 1) >> 1;
    size_t smem_bytes = (size_t)kWarps * packed_per_warp * sizeof(int);

    auto hist_fn = k2q_hist_kernel<kWarps>;
    auto scatter_fn = k2q_scatter_kernel<kWarps>;
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        hist_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        scatter_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    dim3 grid_q(G, B, H);
    hist_fn<<<grid_q, kWarps * kWarpSize, smem_bytes, stream>>>(
        q2k.data_ptr<int>(),
        has_variable_nums ? q2k_nums.data_ptr<int>() : nullptr,
        row_counts.data_ptr<int>(),
        tile_counts.data_ptr<int>(),
        B,
        H,
        Q,
        max_kv,
        num_kv_blocks,
        block_sparse_num,
        has_variable_nums,
        q_per_cta,
        q_per_warp);
    AT_CUDA_CHECK(cudaGetLastError());

    k2q_row_prefix_kernel<1024><<<dim3(B, H), 1024, 0, stream>>>(
        row_counts.data_ptr<int>(),
        has_variable_nums ? bh_offsets.data_ptr<int>() : nullptr,
        row_ptr.data_ptr<int>(),
        schedule_mode == kScheduleModeRowChunk ? schedule_metadata.data_ptr<int>() : nullptr,
        schedule_mode == kScheduleModeRowChunk ? schedule_work_counts.data_ptr<int>() : nullptr,
        B,
        H,
        num_kv_blocks,
        fixed_edges_per_bh,
        has_variable_nums,
        schedule_capacity_per_bh,
        target_q_per_cta);
    AT_CUDA_CHECK(cudaGetLastError());

    constexpr int kPtRowsPerBlock = 8;
    constexpr int kPtThreads = 256;
    int G_total = G * kWarps;
    int blocks_per_bh = (num_kv_blocks + kPtRowsPerBlock - 1) / kPtRowsPerBlock;
    int pt_grid = std::max(1, B * H * blocks_per_bh);
    size_t pt_smem = (size_t)kPtRowsPerBlock * G_total * sizeof(int);
    auto tile_prefix_fn = k2q_tile_prefix_smem_kernel<kPtThreads, kPtRowsPerBlock>;
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        tile_prefix_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)pt_smem));
    tile_prefix_fn<<<pt_grid, kPtThreads, pt_smem, stream>>>(
        tile_counts.data_ptr<int>(),
        row_ptr.data_ptr<int>(),
        B,
        H,
        num_kv_blocks,
        G_total);
    AT_CUDA_CHECK(cudaGetLastError());

    scatter_fn<<<grid_q, kWarps * kWarpSize, smem_bytes, stream>>>(
        q2k.data_ptr<int>(),
        has_variable_nums ? q2k_nums.data_ptr<int>() : nullptr,
        tile_counts.data_ptr<int>(),
        q_idx.data_ptr<int>(),
        B,
        H,
        Q,
        max_kv,
        num_kv_blocks,
        q_idx_capacity,
        block_sparse_num,
        has_variable_nums,
        q_per_cta,
        q_per_warp);
    AT_CUDA_CHECK(cudaGetLastError());

    if (schedule_mode == kScheduleModeQRange) {
        int threads = 256;
        dim3 grid_sched((schedule_capacity_per_bh + threads - 1) / threads, H, B);
        k2q_qrange_schedule_kernel<<<grid_sched, threads, 0, stream>>>(
            row_ptr.data_ptr<int>(),
            q_idx.data_ptr<int>(),
            schedule_metadata.data_ptr<int>(),
            schedule_work_counts.data_ptr<int>(),
            B,
            H,
            num_kv_blocks,
            qrange_num_q_groups,
            qrange_q_bucket_size_blocks,
            schedule_capacity_per_bh);
        AT_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace

void run_build_k2q_csr(
    torch::Tensor q2k,
    torch::Tensor q2k_nums,
    torch::Tensor bh_offsets,
    torch::Tensor row_ptr,
    torch::Tensor q_idx,
    torch::Tensor schedule_metadata,
    torch::Tensor schedule_work_counts,
    int64_t target_q_per_cta,
    int64_t schedule_capacity_per_bh,
    bool emit_schedule,
    int64_t schedule_mode_arg,
    int64_t qrange_num_q_groups,
    int64_t qrange_q_bucket_size_blocks,
    int64_t block_sparse_num,
    int64_t num_kv_blocks,
    int64_t total_edges,
    bool has_variable_nums)
{
    CHECK_INPUT(q2k);
    CHECK_INPUT(row_ptr);
    CHECK_INPUT(q_idx);
    TORCH_CHECK(q2k.dim() == 4, "q2k must have shape [B, H, Q, max_kv]");
    TORCH_CHECK(row_ptr.dim() == 3, "row_ptr must have shape [B, H, num_kv_blocks + 1]");
    TORCH_CHECK(q_idx.dim() == 1, "q_idx must have shape [total_edges]");

    int B = (int)q2k.size(0);
    int H = (int)q2k.size(1);
    int Q = (int)q2k.size(2);
    int max_kv = (int)q2k.size(3);
    int Nkv = (int)num_kv_blocks;
    int bsn = (int)block_sparse_num;
    TORCH_CHECK(total_edges >= 0 && total_edges <= INT_MAX,
                "total_edges must fit int32 CSR offsets");
    TORCH_CHECK(q_idx.size(0) == total_edges, "q_idx total_edges mismatch");
    int q_idx_capacity = (int)q_idx.size(0);
    int schedule_mode = emit_schedule ? (int)schedule_mode_arg : kScheduleModeNone;
    long long fixed_edges_per_bh_ll = (long long)Q * (long long)bsn;
    TORCH_CHECK(fixed_edges_per_bh_ll >= 0 && fixed_edges_per_bh_ll <= INT_MAX,
                "fixed per-BH edge count must fit int32");
    int fixed_edges_per_bh = (int)fixed_edges_per_bh_ll;

    TORCH_CHECK(Nkv >= 0, "num_kv_blocks must be non-negative");
    TORCH_CHECK(row_ptr.size(0) == B && row_ptr.size(1) == H && row_ptr.size(2) == Nkv + 1,
                "row_ptr shape mismatch");
    TORCH_CHECK(q_idx.device() == q2k.device() && row_ptr.device() == q2k.device(),
                "all output tensors must share q2k device");
    if (emit_schedule) {
        CHECK_INPUT(schedule_metadata);
        CHECK_INPUT(schedule_work_counts);
        TORCH_CHECK(schedule_mode == kScheduleModeRowChunk ||
                    schedule_mode == kScheduleModeQRange,
                    "schedule_mode must be 1(row_chunk) or 2(qrange) when emit_schedule=true");
        TORCH_CHECK(target_q_per_cta > 0, "target_q_per_cta must be positive");
        TORCH_CHECK(schedule_capacity_per_bh > 0, "schedule_capacity_per_bh must be positive");
        TORCH_CHECK(schedule_metadata.dim() == 4 &&
                    schedule_metadata.size(0) == B &&
                    schedule_metadata.size(1) == H &&
                    schedule_metadata.size(2) == schedule_capacity_per_bh &&
                    schedule_metadata.size(3) == 4,
                    "schedule_metadata must have shape [B, H, capacity, 4]");
        TORCH_CHECK(schedule_work_counts.dim() == 2 &&
                    schedule_work_counts.size(0) == B &&
                    schedule_work_counts.size(1) == H,
                    "schedule_work_counts must have shape [B, H]");
        TORCH_CHECK(schedule_metadata.device() == q2k.device() &&
                    schedule_work_counts.device() == q2k.device(),
                    "schedule tensors must share q2k device");
        if (schedule_mode == kScheduleModeQRange) {
            TORCH_CHECK(qrange_num_q_groups > 0, "qrange_num_q_groups must be positive");
            TORCH_CHECK(qrange_q_bucket_size_blocks > 0, "qrange_q_bucket_size_blocks must be positive");
            TORCH_CHECK(schedule_capacity_per_bh == qrange_num_q_groups * (int64_t)Nkv,
                        "qrange schedule capacity must equal num_q_groups * num_kv_blocks");
        }
    }
    if (has_variable_nums) {
        CHECK_INPUT(q2k_nums);
        CHECK_INPUT(bh_offsets);
        TORCH_CHECK(q2k_nums.dim() == 3, "q2k_nums must have shape [B, H, Q]");
        TORCH_CHECK(q2k_nums.size(0) == B && q2k_nums.size(1) == H && q2k_nums.size(2) == Q,
                    "q2k_nums shape mismatch");
        TORCH_CHECK(q2k_nums.device() == q2k.device(), "q2k_nums device mismatch");
        TORCH_CHECK(bh_offsets.dim() == 1 && bh_offsets.size(0) == (int64_t)B * H + 1,
                    "bh_offsets must have shape [B * H + 1]");
        TORCH_CHECK(bh_offsets.device() == q2k.device(), "bh_offsets device mismatch");
    } else {
        TORCH_CHECK(bsn >= 0 && bsn <= max_kv, "block_sparse_num out of range");
        TORCH_CHECK(total_edges == (int64_t)B * H * fixed_edges_per_bh,
                    "fixed total_edges mismatch");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_CUDA_CHECK(cudaMemsetAsync(
        row_ptr.data_ptr<int>(), 0, (size_t)B * H * (Nkv + 1) * sizeof(int), stream));
    AT_CUDA_CHECK(cudaMemsetAsync(
        q_idx.data_ptr<int>(), 0xFF, (size_t)q_idx_capacity * sizeof(int), stream));
    if (emit_schedule && schedule_mode == kScheduleModeRowChunk) {
        AT_CUDA_CHECK(cudaMemsetAsync(
            schedule_metadata.data_ptr<int>(), 0,
            (size_t)B * H * (int)schedule_capacity_per_bh * 4 * sizeof(int), stream));
        AT_CUDA_CHECK(cudaMemsetAsync(
            schedule_work_counts.data_ptr<int>(), 0,
            (size_t)B * H * sizeof(int), stream));
    }

    if (B == 0 || H == 0 || Q == 0 || Nkv == 0 || q_idx_capacity == 0) {
        return;
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(q2k.device());
    auto row_counts = torch::zeros({B, H, Nkv}, opts);

    int dev = q2k.get_device();
    int num_sms = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));

    int per_warp_smem = ((Nkv + 1) >> 1) * (int)sizeof(int);
    int kWarps_pick = 4;
    while (kWarps_pick > 1 && (kWarps_pick * per_warp_smem) * 2 > 228 * 1024) {
        kWarps_pick >>= 1;
    }
    if (kWarps_pick < 1) {
        kWarps_pick = 1;
    }

    int per_cta_smem = kWarps_pick * per_warp_smem;
    int max_ctas_per_sm = std::max(1, (228 * 1024) / std::max(1, per_cta_smem));
    max_ctas_per_sm = std::min(max_ctas_per_sm, 8);
    // BSA uses block-level Q, so Q is far smaller than MiniMax's token-level
    // total_q. Use small q ranges per CTA to expose enough parallelism on B200.
    constexpr int kMinQPerCta = 1;
    int target_ctas_per_sm = 2;
    long long q_work_per_bh = (long long)Q * (long long)(has_variable_nums ? max_kv : bsn);
    if (q_work_per_bh < 1000000LL) {
        target_ctas_per_sm = 1;
    }
    if (const char* env = std::getenv("BSA_K2Q_CSR_CTAS_PER_SM")) {
        target_ctas_per_sm = std::max(1, std::atoi(env));
    }
    int target_g = num_sms * std::min(max_ctas_per_sm, target_ctas_per_sm);
    int max_g_for_q = (Q + kMinQPerCta - 1) / kMinQPerCta;
    int G = std::min({std::max(1, target_g), std::max(1, max_g_for_q), std::max(1, Q)});
    int q_per_cta = (Q + G - 1) / G;
    G = (Q + q_per_cta - 1) / q_per_cta;
    int q_per_warp = (q_per_cta + kWarps_pick - 1) / kWarps_pick;
    int G_total = G * kWarps_pick;

    auto tile_counts = torch::empty({G_total, B, H, Nkv}, opts);

    if (kWarps_pick == 4) {
        launch_hist_scatter<4>(
            q2k, q2k_nums, bh_offsets, row_ptr, q_idx,
            schedule_metadata, schedule_work_counts, row_counts, tile_counts,
            B, H, Q, max_kv, Nkv, q_idx_capacity, fixed_edges_per_bh, bsn, has_variable_nums,
            emit_schedule, schedule_mode, (int)schedule_capacity_per_bh, (int)target_q_per_cta,
            (int)qrange_num_q_groups, (int)qrange_q_bucket_size_blocks,
            G, q_per_cta, q_per_warp, stream);
    } else if (kWarps_pick == 2) {
        launch_hist_scatter<2>(
            q2k, q2k_nums, bh_offsets, row_ptr, q_idx,
            schedule_metadata, schedule_work_counts, row_counts, tile_counts,
            B, H, Q, max_kv, Nkv, q_idx_capacity, fixed_edges_per_bh, bsn, has_variable_nums,
            emit_schedule, schedule_mode, (int)schedule_capacity_per_bh, (int)target_q_per_cta,
            (int)qrange_num_q_groups, (int)qrange_q_bucket_size_blocks,
            G, q_per_cta, q_per_warp, stream);
    } else {
        launch_hist_scatter<1>(
            q2k, q2k_nums, bh_offsets, row_ptr, q_idx,
            schedule_metadata, schedule_work_counts, row_counts, tile_counts,
            B, H, Q, max_kv, Nkv, q_idx_capacity, fixed_edges_per_bh, bsn, has_variable_nums,
            emit_schedule, schedule_mode, (int)schedule_capacity_per_bh, (int)target_q_per_cta,
            (int)qrange_num_q_groups, (int)qrange_q_bucket_size_blocks,
            G, q_per_cta, q_per_warp, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "run_build_k2q_csr",
        &run_build_k2q_csr,
        "BSA q2k -> k2q CSR build (runtime topK, sorted rows)",
        pybind11::arg("q2k"),
        pybind11::arg("q2k_nums"),
        pybind11::arg("bh_offsets"),
        pybind11::arg("row_ptr"),
        pybind11::arg("q_idx"),
        pybind11::arg("schedule_metadata"),
        pybind11::arg("schedule_work_counts"),
        pybind11::arg("target_q_per_cta"),
        pybind11::arg("schedule_capacity_per_bh"),
        pybind11::arg("emit_schedule"),
        pybind11::arg("schedule_mode"),
        pybind11::arg("qrange_num_q_groups"),
        pybind11::arg("qrange_q_bucket_size_blocks"),
        pybind11::arg("block_sparse_num"),
        pybind11::arg("num_kv_blocks"),
        pybind11::arg("total_edges"),
        pybind11::arg("has_variable_nums"));
}
