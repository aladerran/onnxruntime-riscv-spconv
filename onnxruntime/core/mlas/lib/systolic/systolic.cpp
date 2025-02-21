// See LICENSE for license details.

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <google/dense_hash_map>
#include <iostream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "systolic_include.h"
#pragma GCC diagnostic pop

/**
 * Perform a matmul and subsequent quantization.
 * Switch between TILED_OS and TILED_CPU
 * 
 * Elements are accumulated internally into acc_t (int32) and subsequently rounded/saturated to elem_t (int8).
 * The given divisor *must* be a power of 2.
 */

#define ROTATED_MATMUL_TYPE(x)

/**
 * Interally CPU is last in tiled_matmul_type_t but we want to expose CPU as accelerator mode 0
 * So just rotate everything by one
 */
inline int positive_mod(int i, int n) {
  return (i % n + n) % n;
}
inline tiled_matmul_type_t get_accelerator_mode(int mode) {
  return static_cast<tiled_matmul_type_t>(positive_mod(mode - 1, (int)CPU + 1));
}

/* Internal -- no need to touch */


/**
 * Wrapper function around tiled_matmul_auto that provides a BLAS like interface
 * C := alpha*op( A )op( B ) + beta*D
 * Note that like blas, dim_I dim_J and dim_K are after the transpose is applied
 * 
 * No output scale is applied, so this is best used with floating point types
 */
void tiled_gemm_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                     size_t strideA,
                     size_t strideB,
                     size_t strideD,
                     size_t strideC,
                     const elem_t* A, const elem_t* B,
                     const acc_t* D, elem_t* C,
                     int act, scale_t scaleAlpha, acc_scale_t scaleBeta, bool repeating_bias,
                     bool transA, bool transB,
                     enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_auto(dim_I, dim_J, dim_K,
                    A, B, D, C,
                    strideA, strideB, strideD, strideC,
                    scaleAlpha, MVIN_SCALE_IDENTITY, scaleBeta,
                    act, ACC_SCALE_IDENTITY, /*relu6_shift -> bert_scale=*/ 0, repeating_bias,
                    transA, transB,
                    /*full_c= */ false, /*low_d= */ false,
                    /*weightA= */ 3,
                    tiled_matmul_type);
}

/**
 * Wrapper function around above tiled_gemm_auto that assumes full stride (equal to matrix width)
 */
void tiled_gemm_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                     const elem_t* A, const elem_t* B,
                     const acc_t* D, elem_t* C,
                     int act, scale_t scaleAlpha, acc_scale_t scaleBeta, bool repeating_bias,
                     bool transA, bool transB,
                     enum tiled_matmul_type_t tiled_matmul_type) {
  int lda = transA ? dim_I : dim_K;
  int ldb = transB ? dim_K : dim_J;
  tiled_gemm_auto(dim_I, dim_J, dim_K, lda, ldb, dim_J, dim_J,
                  A, B, D, C,
                  act, scaleAlpha, scaleBeta, repeating_bias,
                  transA, transB, tiled_matmul_type);
}

/**
 * Wrapper function around tiled_matmul_auto that provides a simple interface to
 * call for matrix/matrix multiplication C = scale*(A*B + D)
 */
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                       size_t strideA,
                       size_t strideB,
                       size_t strideD,
                       size_t strideC,
                       const elem_t* A, const elem_t* B,
                       const acc_t* D, elem_t* C,
                       int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
                       bool transA, bool transB,
                       enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_auto(dim_I, dim_J, dim_K,
                    A, B, D, C,
                    strideA, strideB, strideD, strideC,
                    MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, ACC_SCALE_IDENTITY,
                    act, scale, relu6_shift, repeating_bias,
                    transA, transB,
                    /*full_c= */ false, /*low_d= */ false,
                    /*weightA= */ 3,
                    tiled_matmul_type);
}

/**
 * Wrapper function around above that assumes stride is full matrix width
 */
void tiled_matmul_auto(size_t dim_I, size_t dim_J, size_t dim_K,
                       const elem_t* A, const elem_t* B,
                       const acc_t* D, elem_t* C,
                       int act, acc_scale_t scale, size_t relu6_shift, bool repeating_bias,
                       enum tiled_matmul_type_t tiled_matmul_type) {
  tiled_matmul_auto(dim_I, dim_J, dim_K, dim_K, dim_J, dim_J, dim_J,
                    A, B, D, C,
                    act, scale, relu6_shift, repeating_bias,
                    /*transA= */ false, /*transB= */ false, tiled_matmul_type);
}

/* End internal */

/**
 * An interface similar to Gemmlowp's matrix multiply
 * Does real_multiplier*(in1 * in2 + bias)
 */
void SystolicMultiply(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK,
                      const elem_t* in1, const elem_t* in2, elem_t* out, acc_scale_t real_multiplier, const acc_t* bias) {
#ifndef FOR_FIRESIM
//   printf("Called into systolic matmul!\n");
//   printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
#endif
  tiled_matmul_auto(dimI, dimJ, dimK, in1, in2, bias, out, /*activation= */ relu,
                    real_multiplier,
                    /*relu6_shift= */ 0, /* repeating_bias= */ 0,
                    get_accelerator_mode(accelerator_mode));
}

#ifdef SYSTOLIC_FP32
/**
 * Provides an interface similar to BLAS matrix multiply
 * C = alpha*A*B + beta*C
 */
void SystolicGemm(char accelerator_mode,
                  bool TransA,
                  bool TransB,
                  size_t M,
                  size_t N,
                  size_t K,
                  scale_t alpha,
                  const elem_t* A,
                  const elem_t* B,
                  acc_scale_t beta,
                  elem_t* C) {
#ifndef FOR_FIRESIM
  printf("Called into systolic gemm!\n");
  printf("Using accelerated gemm with dimensions (%zd, %zd, %zd)\n", M, N, K);
#endif

  tiled_gemm_auto(M, N, K, A, B, beta == 0 ? nullptr : C, C, /*activation= */ false,
                  alpha, beta, /* repeating_bias= */ 0,
                  TransA, TransB, get_accelerator_mode(accelerator_mode));
}

void SystolicGemm(char accelerator_mode,
                  bool TransA,
                  bool TransB,
                  size_t M,
                  size_t N,
                  size_t K,
                  scale_t alpha,
                  const elem_t* A,
                  int lda,
                  const elem_t* B,
                  int ldb,
                  acc_scale_t beta,
                  elem_t* C,
                  int ldc) {
#ifndef FOR_FIRESIM
  printf("Called into systolic gemm!\n");
  printf("Using accelerated gemm with dimensions (%zd, %zd, %zd)\n", M, N, K);
#endif
  tiled_gemm_auto(M, N, K,
                 lda, ldb, ldc, ldc,
                 A, B, beta == 0 ? nullptr : C, C, /*activation= */ false,
                  alpha, beta, /* repeating_bias= */ 0,
                  TransA, TransB, get_accelerator_mode(accelerator_mode));
}
#endif

/**
 * Provides a matrix multiply that allows specifying strides
 */
void SystolicMultiply(char accelerator_mode, bool relu,
                      int dimI, int dimJ, int dimK,
                      const elem_t* in1, int strideIn1,
                      const elem_t* in2, int strideIn2,
                      elem_t* out, int strideOut,
                      acc_scale_t real_multiplier,
                      const acc_t* bias, int strideBias, bool repeating_bias) {
#ifndef FOR_FIRESIM
//   printf("Called into systolic matmul!\n");
//   printf("Using accelerated matmul with dimensions (%d, %d, %d)\n", dimI, dimJ, dimK);
#endif
  tiled_matmul_auto(dimI, dimJ, dimK,
                    strideIn1, strideIn2, strideBias, strideOut,
                    in1, in2, bias, out, /*activation= */ relu,
                    real_multiplier, /*relu6_shift= */ 0, /* repeating_bias= */ repeating_bias,
                    /*transA= */ false, /*transB= */ false,
                    get_accelerator_mode(accelerator_mode));
}

/**
 * Adds two matrices elementwise
 */
void SystolicAdd(char accelerator_mode __attribute__((unused)), bool relu, const elem_t* A, float A_scale, const elem_t* B,
                 float B_scale,
                 elem_t* C, float C_scale, int dim) {
#ifndef FOR_FIRESIM
  printf("Called into systolic add\n");
#endif
  // To most efficiently use systolic, instead of using 1xdim, we use 16xResizedDim.
  // Systolic can load multiple blocks in a given row

  // Note that it's more accurate to use A_scale/C_scale and B_scale/C_scale as the A, B scales (with C_scale = 1)
  // Since that way we don't blow up rounding error by dividing by C_scale

  // Equivalent to:
  // for (int i = 0; i < dim; i++) {
  //   int32_t tmp1 = (int) MVIN_SCALE(*A, A_scale/C_scale);
  //   int32_t tmp2 = (int) MVIN_SCALE(*B, B_scale/C_scale);
  //   *C = scale_and_sat(tmp1 + tmp2, relu ? RELU : 0, 1, 0);

  //   A++;
  //   B++;
  //   C++;
  // }

  int resizedDim = dim - dim % DIM;
  tiled_resadd_auto(DIM, resizedDim / DIM, A_scale / C_scale, B_scale / C_scale,
                    /*C_scale= */ 1, A, B, C, relu, get_accelerator_mode(accelerator_mode));
  if (dim % DIM > 0) {
#ifndef FOR_FIRESIM
    printf("Some extra leftover\n");
#endif
    tiled_resadd_auto(1, dim % DIM, A_scale / C_scale, B_scale / C_scale,
                      /*C_scale= */ 1, A + resizedDim, B + resizedDim, C + resizedDim, relu, get_accelerator_mode(accelerator_mode));
  }
}

/**
 * Convolution of two matrices. Input must be in NHWC format, weight must be in HWIO format
 */
void SystolicConv(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const elem_t* input,
                  const elem_t* weights,
                  const acc_t* bias,
                  elem_t* output,
                  bool relu,
                  float output_scale,
                  int pool_size = 0, int pool_stride = 0, int pool_padding = 0) {
  printf("Called into systolic conv\n");
  if (pool_size != 0) {
    printf("Using systolic pooling\n");
  }
  // printf("Debugging info\n");
  // printf("Batch size, in_w/h, in_channel %d %d %d\n", batch_size, in_dim, in_channels);
  // printf("Out_channels, out_w/h %d %d\n", out_channels, out_dim);
  // printf("Stride, padding %d %d\n", stride, padding);
  // printf("kernel_w/h %d\n", kernel_dim);
  // if (bias) {
  //   printf("Bias values: %d\n", bias[0]);
  // }
  // printf("Relu? %d\n", relu);

  tiled_conv_auto(batch_size, in_dim, in_channels, out_channels, out_dim,
                  stride,
                  /*input_dilation= */ 1,
                  /*kernel_dilation= */ 1,
                  padding, kernel_dim,
                  /*wrot180= */ false, 
                  /*trans_output_1203= */ false,
                  /*trans_input_3120= */ false,
                  /*trans_weight_1203= */ false,
                  /*trans_weight_0132= */ false,
                  input, weights, bias, output,
                  relu, output_scale, /*relu6_shift deprecated*/
                  pool_size, pool_stride, pool_padding,
                  get_accelerator_mode(accelerator_mode));

  // printf("Output\n");
  // for (int i = 0; i < out_dim * out_dim * out_channels * batch_size; i++) {
  //   printf("%d ", output[i]);
  // }
  // printf("\n");
}

void SystolicConvTranspose(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const elem_t* input,
                  const elem_t* weights,
                  const acc_t* bias,
                  elem_t* output,
                  bool relu,
                  float output_scale) {
  printf("Called into systolic conv transpose\n");


  tiled_conv_auto(batch_size, in_dim, in_channels, out_channels, out_dim,
                  /*stride = */ 1,
                  /*input_dilation= */ stride,
                  /*kernel_dilation= */ 1,
                  /*padding= */ kernel_dim - 1 - padding,
                  kernel_dim,
                  /*wrot180= */ true,
                  /*trans_output_1203= */ false,
                  /*trans_input_3120= */ false,
                  /*trans_weight_1203= */ false,
                  /*trans_weight_0132= */ true,
                  input, weights, bias, output,
                  relu, output_scale, /*relu6_shift deprecated*/
                  0, 0, 0,
                  get_accelerator_mode(accelerator_mode));
}


/**
 * Note that the batch size and dimensions are _after_ transposition is applied
 */
void SystolicConvBackpropFilter(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const elem_t* input,
                  const elem_t* weights,
                  const acc_t* bias,
                  elem_t* output,
                  bool relu,
                  float output_scale) {
  printf("Called into systolic conv backprop filter\n");


  tiled_conv_auto(batch_size, in_dim, in_channels, out_channels, out_dim,
                  /*stride = */ 1,
                  /*input_dilation= */ 1,
                  /*kernel_dilation= */ stride,
                  /*padding= */ padding,
                  kernel_dim,
                  /*wrot180= */ false, 
                  /*trans_output_1203= */ true,
                  /*trans_input_3120= */ true,
                  /*trans_weight_1203= */ true,
                  /*trans_weight_0132= */ false,
                  input, weights, bias, output,
                  relu, output_scale, /*relu6_shift deprecated*/
                  0, 0, 0,
                  get_accelerator_mode(accelerator_mode));
}

// We do this to clear out gemmini on every process launch
#ifdef FOR_FIRESIM
__attribute__((constructor))
void cleargemmini() {
  gemmini_flush(0);
}
#endif

unsigned long long read_cycles()
{
    unsigned long long cycles;
    asm volatile ("rdcycle %0" : "=r" (cycles));
    return cycles;
}

/**
 * Systolic sparse convolution backend, CPU implementation also included
 ==================================================================================================
 */


/**
 * Adds two matrices elementwise
 */
void SystolicAdd_FP32(char accelerator_mode __attribute__((unused)), bool relu, const elem_t* A, float A_scale, const elem_t* B,
                 float B_scale,
                 elem_t* C, float C_scale, int dim) {
#ifndef FOR_FIRESIM
//   printf("Called into systolic add (FP32)\n");
  if (relu) {
    // printf("Called into systolic relu\n");
  }
#endif

  int resizedDim = dim - dim % DIM;
  tiled_resadd_auto(DIM, resizedDim / DIM, A_scale / C_scale, B_scale / C_scale,
                    /*C_scale= */ 1, A, B, C, relu, get_accelerator_mode(accelerator_mode));
  if (dim % DIM > 0) {
#ifndef FOR_FIRESIM
    printf("Some extra leftover\n");
#endif
    tiled_resadd_auto(1, dim % DIM, A_scale / C_scale, B_scale / C_scale,
                      /*C_scale= */ 1, A + resizedDim, B + resizedDim, C + resizedDim, relu, get_accelerator_mode(accelerator_mode));
  }
}


unsigned long long gather_cycles;
unsigned long long scatter_cycles;
unsigned long long matmul_cycles;
unsigned long long buffer_cycles;
unsigned long long zeroing_cycles;
// unsigned long long gather_matmul_scatter_cycles;
unsigned long long hash_cycles;
unsigned long long hash_kernel_cycles;
unsigned long long hash_query_cycles;


// Gemmini matmul
void matmul_gemmini(const float *A, const float *B, float *C, int M, int N, int K,
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

void matmul_type_dispatch(tiled_matmul_type_t tiled_matmul_type, 
                          const float* A, const float* B, float* C, 
                          int M, int N, int K) {

    auto matmul_start = read_cycles();
    
    switch (tiled_matmul_type) {
        case CPU:
            matmul_gemmini(A, B, C, M, N, K, CPU);
            break;
        case OS:
            // std::cout << "Using Gemmini OS Matmul!" << std::endl;
            matmul_gemmini(A, B, C, M, N, K, OS);
            break;
        case WS:
            // std::cout << "Using Gemmini WS Matmul!" << std::endl;
            matmul_gemmini(A, B, C, M, N, K, WS);
            break;
        default:
            throw std::invalid_argument("Invalid matmul type");
    }

    matmul_cycles += read_cycles() - matmul_start;

    // std::cout << "Using accelerated matmul with dimensions (" << M << ", " << N << ", " << K << ")" << std::endl;

}

void scatter_cpu(const int n_in, const int n_out, const int c,
                 const float *in_feat, float *out_feat, const int *kmap,
                 const bool transpose) {

    // std::cout << "Scattering " << n_in << " points" << std::endl;

    auto scatter_start = read_cycles();

    for (int i = 0; i < n_in; i++) {
        int out_pos = kmap[2 * i + 1 - transpose];
        if (out_pos < 0) {
            continue;
        }
        for (int j = 0; j < c; j++) {
            out_feat[out_pos * c + j] += in_feat[i * c + j];
        }
    }

    scatter_cycles += read_cycles() - scatter_start;

}

void gather_cpu(const int n_k, const int n_in, const int c,
                const float *in_feat, float *out_feat, const int *kmap,
                const bool transpose) {

    // std::cout << "Gathering " << n_k << " points" << std::endl;

    auto gather_start = read_cycles();

    for (int i = 0; i < n_k; i++) {
        int in_pos = kmap[2 * i + transpose];
        if (in_pos < 0) {
            continue;
        }
        for (int j = 0; j < c; j++) {
            out_feat[i * c + j] = in_feat[in_pos * c + j];
        }
    }

    gather_cycles += read_cycles() - gather_start;

}

void convolution_forward_cpu(const float* in_feat,
                             float* out_feat,
                             const float* kernel,
                             const int* neighbor_map,
                             const int* neighbor_offset,
                             const bool transpose,
                             const int in_channels,
                             const int out_channels,
                             const int in_nrows,
                             const int out_nrows,
                             const int kernel_volume,
                             char accelerator_mode) {


    tiled_matmul_type_t tiled_matmul_type = get_accelerator_mode(accelerator_mode);

    auto zeoring_start = read_cycles();

    // Initialize output features to zero
    std::fill(out_feat, out_feat + out_nrows * out_channels, 0);

    zeroing_cycles += read_cycles() - zeoring_start;

    int in_buffer_size = 1;
    bool flag = false;

    // Determine buffer size for memory optimization
    if (kernel_volume % 2 && out_nrows == in_nrows) {
        flag = true;
        in_buffer_size =
            *std::max_element(neighbor_offset,
                              neighbor_offset + kernel_volume / 2);
        in_buffer_size =
            std::max(in_buffer_size,
                     *std::max_element(
                         neighbor_offset + kernel_volume / 2 + 1,
                         neighbor_offset + kernel_volume));
        in_buffer_size = std::max(in_buffer_size, 1);

        // Perform initial matrix multiplication
        matmul_type_dispatch(tiled_matmul_type,
                             in_feat,
                             kernel + (kernel_volume / 2) * in_channels * out_channels,
                             out_feat,
                             in_nrows,
                             out_channels,
                             in_channels);
    } else {
        in_buffer_size =
            *std::max_element(neighbor_offset,
                              neighbor_offset + kernel_volume);
    }

    // std::cout << "in_buffer_size: " << in_buffer_size << std::endl;

    auto buffer_start = read_cycles();

    std::vector<float> in_buffer(in_buffer_size * in_channels, 0);
    std::vector<float> out_buffer(in_buffer_size * out_channels, 0);

    buffer_cycles += read_cycles() - buffer_start;

    int cur_offset = 0;

    for (int i = 0; i < kernel_volume; i++) {

        if (flag && (i == kernel_volume / 2)) {
            cur_offset += 2 * neighbor_offset[i];
            continue;
        }

        if (neighbor_offset[i] == 0) {
            continue;
        }

        // Gather
        gather_cpu(neighbor_offset[i], in_nrows, in_channels,
                   in_feat, in_buffer.data(),
                   neighbor_map + cur_offset, transpose);

        // Matrix multiplication
        matmul_type_dispatch(tiled_matmul_type,
                             in_buffer.data(),
                             kernel + i * in_channels * out_channels,
                             out_buffer.data(),
                             neighbor_offset[i],
                             out_channels,
                             in_channels);

        // Scatter
        scatter_cpu(neighbor_offset[i], out_nrows, out_channels,
                    out_buffer.data(),
                    out_feat,
                    neighbor_map + cur_offset, transpose);
        cur_offset += 2 * neighbor_offset[i];

    }

}

// // TODO: combine gather-matmul-scatter into one function to reduce memory footprint
// // what we are trying to do here is to gather and scatter feats directly to and to from spad/acc without using buffers
// void tiled_gather_matmul_scatter(const int n_k, const int in_nrows, const int in_channels,
//                                  const float* in_feat, float* out_feat,
//                                  const float* kernel, const int out_channels,
//                                  const int* kmap, const bool transpose, 
//                                  const int buffer_size, const int out_nrows, 
//                                  char accelerator_mode) {

//     auto gather_matmul_scatter_start = read_cycles();

//     auto in_buffer_size = buffer_size * in_channels;
//     auto out_buffer_size = buffer_size * out_channels;

//     const uint32_t A_sp_addr_start = 0;
//     const uint32_t B_sp_addr_start = BANK_NUM * BANK_ROWS - out_buffer_size;
//     // const uint32_t C_sp_addr_start = 

//     tiled_matmul_type_t tiled_matmul_type = get_accelerator_mode(accelerator_mode);

//     gemmini_config_ld(sizeof(elem_t));

//     for (int i = 0; i < n_k; i++) {
//         int in_pos = kmap[2 * i + transpose];
//         if (in_pos < 0) {
//             continue;
//         }
//         for (int j = 0; j < in_channels; j++) {
//             // mvin feats to gemmini spad
//             gemmini_mvin(in_feat + in_pos * in_channels + j, j * sizeof(elem_t));
//         }
//     }

//     // placeholder for mvin kernel to gemmini spad
    

//     // placeholder for gemmini matmul logic
    

//     for (int i = 0; i < n_k; i++) {
//         int out_pos = kmap[2 * i + 1 - transpose];
//         if (out_pos < 0) {
//             continue;
//         }
//         for (int j = 0; j < out_channels; j++) {
//             // mvout feats from gemmini acc to out_feat
//             // gemmini_mvout();
//         }
//     }

//     gather_matmul_scatter_cycles += read_cycles() - gather_matmul_scatter_start;
// }

// void convolution_forward_gemmini(const float* in_feat,
//                                           float* out_feat,
//                                           const float* kernel,
//                                           const int* neighbor_map,
//                                           const int* neighbor_offset,
//                                           const bool transpose,
//                                           const int in_channels,
//                                           const int out_channels,
//                                           const int in_nrows,
//                                           const int out_nrows,
//                                           const int kernel_volume,
//                                           char accelerator_mode) {

//     auto gemmini_buffer_start = read_cycles();

//     tiled_matmul_type_t tiled_matmul_type = get_accelerator_mode(accelerator_mode);

//     // Initialize output features to zero ?? maybe not needed if we use gemmini mvout w\ DMA st. ??
//     std::fill(out_feat, out_feat + out_nrows * out_channels, 0);

//     int buffer_size = 1;
//     bool flag = false;

//     // Determine buffer size for memory optimization
//     if (kernel_volume % 2 && out_nrows == in_nrows) {
//         flag = true;
//         buffer_size =
//             *std::max_element(neighbor_offset,
//                               neighbor_offset + kernel_volume / 2);
//         buffer_size =
//             std::max(buffer_size,
//                      *std::max_element(
//                          neighbor_offset + kernel_volume / 2 + 1,
//                          neighbor_offset + kernel_volume));
//         buffer_size = std::max(buffer_size, 1);

//         // Perform initial matrix multiplication
//         matmul_type_dispatch(tiled_matmul_type,
//                              in_feat,
//                              kernel + (kernel_volume / 2) * in_channels * out_channels,
//                              out_feat,
//                              in_nrows,
//                              out_channels,
//                              in_channels);
//     } else {
//         buffer_size =
//             *std::max_element(neighbor_offset,
//                               neighbor_offset + kernel_volume);
//     }

//     int cur_offset = 0;

//     for (int i = 0; i < kernel_volume; i++) {

//         if (flag && (i == kernel_volume / 2)) {
//             cur_offset += 2 * neighbor_offset[i];
//             continue;
//         }

//         if (neighbor_offset[i] == 0) {
//             continue;
//         }

//         tiled_gather_matmul_scatter(neighbor_offset[i], in_nrows, in_channels,
//                                     in_feat, out_feat,
//                                     kernel + i * in_channels * out_channels, out_channels,
//                                     neighbor_map + cur_offset, transpose,
//                                     buffer_size, out_nrows,
//                                     tiled_matmul_type);

//         cur_offset += 2 * neighbor_offset[i];

//         buffer_cycles += read_cycles() - gemmini_buffer_start;
//     }

//     buffer_cycles += read_cycles() - gemmini_buffer_start;

// }


void cpu_hash_wrapper(const int N, const int *data, int64_t *out) {
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

void cpu_hash_32bit_wrapper(const int N, const int *data, uint32_t *out) {
    for (int i = 0; i < N; i++) {
        uint32_t hash = 2166136261U; // 32-bit FNV offset basis
        for (int j = 0; j < 3; j++) {
            hash ^= (unsigned int)data[4 * i + j];
            hash *= 16777619U; // 32-bit FNV prime
        }
        hash = (hash >> 28) ^ (hash & 0xFFFFFFF);
        out[i] = hash;
    }
}

void hash_cpu(const int *idx, int64_t *out, const int N) {
    auto hash_start = read_cycles();
    cpu_hash_wrapper(N, idx, out);
    hash_cycles += read_cycles() - hash_start;
}

void hash_cpu_uint32t(const int *idx, uint32_t *out, const int N) {
    // std::cout << "Hashing " << N << " points" << std::endl;
    auto hash_start = read_cycles();
    cpu_hash_32bit_wrapper(N, idx, out);
    hash_cycles += read_cycles() - hash_start;
}


void cpu_kernel_hash_wrapper(const int N, const int K, const int *data,
                             const int *kernel_offset, int64_t *out) {
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

void cpu_kernel_hash_32bit_wrapper(const int N, const int K, const int *data,
                                   const int *kernel_offset, uint32_t *out) {
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < N; i++) {
            int cur_coord[3];
            for (int j = 0; j < 3; j++) {
                cur_coord[j] = data[i * 4 + j] + kernel_offset[k * 3 + j];
            }
            uint32_t hash = 2166136261U; // 32-bit FNV offset basis
            for (int j = 0; j < 3; j++) {
                hash ^= (unsigned int)cur_coord[j];
                hash *= 16777619U; // 32-bit FNV prime
            }
            hash = (hash >> 28) ^ (hash & 0xFFFFFFF);
            out[k * N + i] = hash;
        }
    }
}

void kernel_hash_cpu(const int *idx, const int *kernel_offset,
                     int64_t *out, const int N, const int K) {
    auto hash_start = read_cycles();
    cpu_kernel_hash_wrapper(N, K, idx, kernel_offset, out);
    hash_kernel_cycles += read_cycles() - hash_start;
}

void kernel_hash_cpu_uint32t(const int *idx, const int *kernel_offset,
                             uint32_t *out, const int N, const int K) {
    // std::cout << "Hashing " << N << " (kernel)points" << std::endl;                            
    auto hash_start = read_cycles();
    cpu_kernel_hash_32bit_wrapper(N, K, idx, kernel_offset, out);
    hash_kernel_cycles += read_cycles() - hash_start;
}


void hash_query_cpu(const int64_t* hash_query, const int64_t* hash_target,
                    const int64_t* idx_target, int64_t* out, const int n, const int n1) {
    auto hash_query_start = read_cycles();
    google::dense_hash_map<int64_t, int64_t> hashmap;
    hashmap.set_empty_key(0);
    
    for (int idx = 0; idx < n; idx++) {
        int64_t key = hash_target[idx];
        int64_t val = idx_target[idx] + 1;
        hashmap.insert(std::make_pair(key, val));
    }
    for (int idx = 0; idx < n1; idx++) {
        int64_t key = hash_query[idx];
        google::dense_hash_map<int64_t, int64_t>::iterator iter = hashmap.find(key);
        if (iter != hashmap.end()) {
            out[idx] = iter->second;
        }
    }
    hash_query_cycles += read_cycles() - hash_query_start;
}


void hash_query_cpu_uint32t(const uint32_t* hash_query, const uint32_t* hash_target,
                            const uint32_t* idx_target, uint32_t* out, const int n, const int n1) {
    auto hash_query_start = read_cycles();
    google::dense_hash_map<uint32_t, uint32_t> hashmap;
    hashmap.set_empty_key(0);
    
    for (int idx = 0; idx < n; idx++) {
        uint32_t key = hash_target[idx];
        uint32_t val = idx_target[idx] + 1;
        hashmap.insert(std::make_pair(key, val));
    }
    for (int idx = 0; idx < n1; idx++) {
        uint32_t key = hash_query[idx];
        google::dense_hash_map<uint32_t, uint32_t>::iterator iter = hashmap.find(key);
        if (iter != hashmap.end()) {
            out[idx] = iter->second;
        }
    }
    hash_query_cycles += read_cycles() - hash_query_start;
}


void print_cycles_backend() {
    std::cout << "Scatter cycles: " << scatter_cycles << std::endl;
    std::cout << "Matmul cycles: " << matmul_cycles << std::endl;
    std::cout << "Gather cycles: " << gather_cycles << std::endl;
    std::cout << "Zeroing cycles: " << zeroing_cycles << std::endl;
    std::cout << "Buffer cycles: " << buffer_cycles << std::endl;
    // std::cout << "Gather-Matmul-Scatter cycles: " << gather_matmul_scatter_cycles << std::endl;
    std::cout << "Hash cycles: " << hash_cycles << std::endl;
    std::cout << "Hash kernel cycles: " << hash_kernel_cycles << std::endl;
    std::cout << "Hash query cycles: " << hash_query_cycles << std::endl;
}
