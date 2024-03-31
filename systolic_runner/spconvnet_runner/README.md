# Submanifold Sparse Convolutional Network (SSCN) Runner

## SSCN Profiling Preview on CPU
The inference time of the SOTA point cloud engine- Torchsparse, tested on my R7-4800hs CPU:
```
(spnn) harb@ubuntu:~/torchsparse$ python examples/backbones.py 
SparseResUNet42:
Execution Time: 228.57015900001443 ms
output[0].F.shape = torch.Size([1000, 32])
output[1].F.shape = torch.Size([1000, 32])
output[2].F.shape = torch.Size([1000, 64])
output[3].F.shape = torch.Size([1000, 128])
output[4].F.shape = torch.Size([1000, 256])
output[5].F.shape = torch.Size([1000, 256])
output[6].F.shape = torch.Size([1000, 128])
output[7].F.shape = torch.Size([1000, 96])
output[8].F.shape = torch.Size([1000, 96])
total_offsets_time: 1.35ms, Count: 9
total_hash_time: 35.63ms, Count: 18
total_query_time: 9.95ms, Count: 9
total_kmap_time: 52.42ms, Count: 9
total_conv3d_forward_time: 119.43ms, Count: 42
total_norm_time: 27.70ms, Count: 49
total_relu_time: 10.73ms, Count: 42
total_downsample_time: 0.68ms, Count: 4
Gather time: 2.45007 ms
Matmul time: 51.2278 ms
Scatter time: 1.34473 ms
(spnn) harb@ubuntu:~/torchsparse$ vi examples/backbones.py 
(spnn) harb@ubuntu:~/torchsparse$ python examples/backbones.py 
SparseResUNet42:
Execution Time: 602.663477999954 ms
output[0].F.shape = torch.Size([10000, 32])
output[1].F.shape = torch.Size([10000, 32])
output[2].F.shape = torch.Size([10000, 64])
output[3].F.shape = torch.Size([10000, 128])
output[4].F.shape = torch.Size([10000, 256])
output[5].F.shape = torch.Size([10000, 256])
output[6].F.shape = torch.Size([10000, 128])
output[7].F.shape = torch.Size([10000, 96])
output[8].F.shape = torch.Size([10000, 96])
total_offsets_time: 2.06ms, Count: 9
total_hash_time: 4.72ms, Count: 18
total_query_time: 35.34ms, Count: 9
total_kmap_time: 56.44ms, Count: 9
total_conv3d_forward_time: 409.71ms, Count: 42
total_norm_time: 44.46ms, Count: 49
total_relu_time: 12.58ms, Count: 42
total_downsample_time: 2.10ms, Count: 4
Gather time: 78.2703 ms
Matmul time: 99.2837 ms
Scatter time: 44.4033 ms
```

When the batch size is small (1k), matrix multiplication dominates the conv_forward runtime; while with a large batch size (10k), data movement dominates. Allocating memory buffer (not shown) also takes up a lot, which is certainly a TODO for us.

Other operations like hashing in the map-building section and non-SpConv layers are slow on CPU, which are major when input size is small.

Our logic for speeding up is:

1. Offload matrix multiplication to the Gemmini Systolic Array (SA).
2. Vectorize hashing operations with the Ara (RVV extension).
3. Implement BN & ReLU layers with SA (also operator fusion).
4. Data movement (the hardest one since we are dealing with a single-thread environment).

Hopefully, we can achieve relatively optimal speedup when processing small batch size data.

## Idea

### 1. Matrix Multiplication

Speeding up matmul is obvious. What we can do here is just some DSE about SA size and Spad/Acc size.

### 2. Hashing Operation

Using the RVV extension is suitable for hashing operations with multiple data. See our implementations with Ara here: [ara/apps].

[ara/apps]:https://github.com/aladerran/ara/tree/main/apps

### 3. BatchNorm Layer

In ther inference session, ignoring \(\epsilon\) and assuming variance (\(\sigma^2\), i.e., `var`) is not zero, the relationship between output \(Y\) and input \(X\) can be simplified to:

$$ Y = \frac{\gamma (X - \mu)}{\sqrt{\sigma^2}} + \beta $$

Where:
- \(X\) is the input data.
- \(Y\) is the output data after batch normalization.
- \(\gamma\) is the `scale` scaling factor.
- \(\beta\) is the `B` bias term.
- \(\mu\) is the `mean`, the moving average mean computed during the training phase for each channel.
- \(\sigma^2\) is the `var`, the moving average variance computed during the training phase for each channel.

To directly compute the batch-normalized \(Y\) using a Systolic Array, we can consider the process as a combination of two main steps: a linear transformation step and an addition step. Specifically:

1. **Linear Transformation**: First, the input \(X\) needs to be centered and scaled, represented as \(\frac{X - \mu}{\sqrt{\sigma^2}}\). In a Systolic Array, this operation can be performed by pre-computing a transformation matrix containing all parameters needed for scaling and centering.

2. **Applying \(\gamma\) and \(\beta\)**: Then, multiply the result from the previous step by \(\gamma\) and add \(\beta\), completing the batch normalization. This step can also be executed on a Systolic Array, treating it as a linear transformation applied to each output element.

Practical steps  include:

- **Preprocessing**: Compute an "extended" matrix \(A\), where each element \(a_{i,j}\) is pre-calculated based on the corresponding \(\gamma\), \(\mu\), and \(\sigma^2\) so that \(X\) can be directly multiplied by it.
- **Matrix Multiplication**: Use the Systolic Array to perform matrix multiplication between \(X\) and \(A\). The parallel nature of the Systolic Array is particularly suited for accelerating this type of operation.
- **Addition**: Add the result to \(\beta\), which can also be performed in parallel in the Systolic Array, with each addition unit handling an output element.

### 4. Operator Fusion

1. BN + ReLU
2. Add + ReLU

---
TBW..
