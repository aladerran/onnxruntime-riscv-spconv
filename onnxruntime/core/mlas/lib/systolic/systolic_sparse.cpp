#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <google/dense_hash_map>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "systolic_include.h"


// Naive matmul
void slow_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < K; ++j) {
            float sum = 0.0;
            for(int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// Gemmini matmul
void gemmini_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                    tiled_matmul_type_t tiled_matmul_type){

    tiled_matmul_auto((size_t)M, (size_t)N, (size_t)K, 
                    (elem_t*)A, (elem_t*)B, 
                    NULL, C, 
                    (size_t)K, (size_t)N, (size_t)N, (size_t)N, 
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
                    NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, false,
                    false, false,
                    false, 0,
                    0,
                    tiled_matmul_type);
}


void scatter_cpu(const int n_in, const int n_out, const int c,
                 const float *in_feat, float *out_feat, const int *kmap,
                 const bool transpose) {
    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i + 1 - transpose];
        if (out_pos < 0) {
            continue;
        }
        for (int j = 0; j < c; j++) {
            out_feat[out_pos * c + j] += in_feat[i * c + j];
        }
    }
}

void gather_cpu(const int n_k, const int n_in, const int c,
                const float *in_feat, float *out_feat, const int *kmap,
                const bool transpose) {
    for (int i = 0; i < n_k; i++) {
        int in_pos = kmap[2 * i + transpose];
        if (in_pos < 0) {
            continue;
        }
        for (int j = 0; j < c; j++) {
            out_feat[i * c + j] = in_feat[in_pos * c + j];
        }
    }
}


void convolution_forward_cpu(const float *in_feat, float *out_feat,
                             const float *kernel, const int *neighbor_map,
                             const int *neighbor_offset, const bool transpose,
                             const int in_nrows, const int out_nrows,
                             const int kernel_volume, const int c, 
                             enum tiled_matmul_type_t tiled_matmul_type) {

    // Initialize output feature with zeros
    std::fill(out_feat, out_feat + out_nrows * c, 0.0f);

    int in_buffer_size =
        *std::max_element(neighbor_offset, neighbor_offset + kernel_volume);

    std::vector<float> in_buffer(in_buffer_size * c, 0.0f);
    std::vector<float> out_buffer(in_buffer_size * c, 0.0f);
    int cur_offset = 0;
    for (int i = 0; i < kernel_volume; i++) {

        if (neighbor_offset[i] == 0) {
            continue;
        }

        float *out_buffer_activated = &out_buffer[0];
        float *in_buffer_activated = &in_buffer[0];

        // gather
        gather_cpu(neighbor_offset[i], in_nrows, c,
                   in_feat, in_buffer_activated,
                   neighbor_map + cur_offset, transpose);

        // matmul
        if (tiled_matmul_type == CPU){
            slow_matmul(in_buffer_activated, kernel + i * c * c, out_buffer_activated, neighbor_offset[i], c, c);
        }
        else{
            gemmini_matmul(in_buffer_activated, kernel + i * c * c, out_buffer_activated, neighbor_offset[i], c, c, tiled_matmul_type);
        }
        // scatter
        scatter_cpu(neighbor_offset[i], out_nrows, c,
                    out_buffer_activated, out_feat,
                    neighbor_map + cur_offset, transpose);
        cur_offset += 2 * neighbor_offset[i];
    }
}


void cpu_hash_wrapper(int N, const int* data, int64_t* out) {
    for (int i = 0; i < N; i++) {
        uint64_t hash = 14695981039346656037UL;
        for (int j = 0; j < 4; j++) {
            hash ^= (unsigned int)data[4 * i + j];
            hash *= 1099511628211UL;
        }
        hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
        out[i] = hash;
    }
}


void cpu_kernel_hash_wrapper(size_t N, int K, const int* data,
                             const int* kernel_offset, int64_t* out) {
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < N; i++) {
            int cur_coord[4];
            for (int j = 0; j < 3; j++) {
                cur_coord[j] = data[i * 4 + j] + kernel_offset[k * 3 + j];
            }
            cur_coord[3] = data[i * 4 + 3];
            uint64_t hash = 14695981039346656037UL;
            for (int j = 0; j < 4; j++) {
                hash ^= (unsigned int)cur_coord[j];
                hash *= 1099511628211UL;
            }
            hash = (hash >> 60) ^ (hash & 0xFFFFFFFFFFFFFFF);
            out[k * N + i] = hash;
        }
    }
}


std::vector<int64_t> hash_cpu(const std::vector<int>& idx) {
    size_t N = idx.size();
    std::vector<int64_t> out(N);
    cpu_hash_wrapper(N, idx.data(), out.data());
    return out;
}


std::vector<int64_t> kernel_hash_cpu(const std::vector<int>& idx,
                                     const std::vector<int>& kernel_offset) {
    int N = idx.size();
    int K = kernel_offset.size() / 3;
    std::vector<int64_t> out(K * N);
    cpu_kernel_hash_wrapper(N, K, idx.data(), kernel_offset.data(), out.data());
    return out;
}


std::vector<int64_t> hash_query_cpu(const std::vector<int64_t>& hash_query,
                                    const std::vector<int64_t>& hash_target,
                                    const std::vector<int64_t>& idx_target) {
    int n = hash_target.size();
    int n1 = hash_query.size();

    google::dense_hash_map<int64_t, int64_t> hashmap;
    hashmap.set_empty_key(0);
    std::vector<int64_t> out(n1, 0);
    
    for (int idx = 0; idx < n; idx++) {
        int64_t key = hash_target[idx];
        int64_t val = idx_target[idx] + 1;
        hashmap.insert(std::make_pair(key, val));
    }
    for (int idx = 0; idx < n1; idx++) {
        int64_t key = hash_query[idx];
        auto iter = hashmap.find(key);
        if (iter != hashmap.end()) {
            out[idx] = iter->second;
        }
    }

    return out;
}

#pragma GCC diagnostic pop
