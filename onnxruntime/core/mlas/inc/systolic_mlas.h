#ifdef SYSTOLIC_INT8

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2,
         int8_t* out, float real_multiplier, const int32_t* bias = nullptr);

void SystolicGemm
MLASCALL(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK, const int8_t* in1, const int8_t* in2,
         int8_t* out, float alpha, float beta, bool transA, bool transB, const int32_t* bias = nullptr);

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu,
                            int dimI, int dimJ, int dimK,
                            const int8_t* in1, int strideIn1,
                            const int8_t* in2, int strideIn2,
                            int8_t* out, int strideOut,
                            float real_multiplier,
                            const int32_t* bias, int strideBias, bool repeating_bias);

void SystolicAdd
MLASCALL(char accelerator_mode, bool relu, const int8_t* in1, float in1_scale, const int8_t* in2,
         float in2_scale,
         int8_t* out, float out_scale, int dim);

void SystolicConv
MLASCALL(char accelerator_mode, int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,
        const int8_t* input,
        const int8_t* weights,
        const int32_t* bias,
        int8_t* output,
        bool relu,
        float output_scale,
        int pool_size = 0, int pool_stride = 0, int pool_padding = 0);

#endif

#ifdef SYSTOLIC_FP32
#include <vector>

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu, int dimI, int dimJ, int dimK, const float* in1, const float* in2,
         float* out, float real_multiplier, const float* bias = nullptr);

void SystolicMultiply
MLASCALL(char accelerator_mode, bool relu,
                            int dimI, int dimJ, int dimK,
                            const float* in1, int strideIn1,
                            const float* in2, int strideIn2,
                            float* out, int strideOut,
                            float real_multiplier,
                            const float* bias, int strideBias, bool repeating_bias);

// Matches Blass Gemm signature 
void SystolicGemm(
   char accelerator_mode, 
    bool TransA,
    bool TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C);

void SystolicGemm(
   char accelerator_mode, 
    bool TransA,
    bool TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    float beta,
    float* C,
    int ldc);

void SystolicConv
MLASCALL(char accelerator_mode, int batch_size, int in_dim, int in_channels,
        int out_channels, int out_dim,
        int stride, int padding, int kernel_dim,
        const float* input,
        const float* weights,
        const float* bias,
        float* output,
        bool relu,
        float output_scale,
        int pool_size = 0, int pool_stride = 0, int pool_padding = 0);

void SystolicConvTranspose(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const float* input,
                  const float* weights,
                  const float* bias,
                  float* output,
                  bool relu,
                  float output_scale);

void SystolicConvBackpropFilter(char accelerator_mode, int batch_size, int in_dim, int in_channels,
                  int out_channels, int out_dim,
                  int stride, int padding, int kernel_dim,
                  const float* input,
                  const float* weights,
                  const float* bias,
                  float* output,
                  bool relu,
                  float output_scale);


// Add sparse conv backend

void convolution_forward_cpu
MLASCALL(const float* in_feat, float* out_feat,
          const float* kernel, const int* neighbor_map,
          const int* neighbor_offset, const bool transpose,
          const int in_channels, const int out_channels,
          const int in_nrows, const int out_nrows,
          const int kernel_volume, char accelerator_mode);

void hash_cpu
MLASCALL(const int *idx, int64_t *out, const int N);

void kernel_hash_cpu
MLASCALL(const int *idx, const int *kernel_offset,
          int64_t *out, const int N, const int K);

void hash_query_cpu
MLASCALL(const int64_t* hash_query, const int64_t* hash_target,
          const int64_t* idx_target, int64_t* out, const int n, const int n1);

void print_cycles
MLASCALL();

unsigned long long read_cycles
MLASCALL();
                            
#endif
