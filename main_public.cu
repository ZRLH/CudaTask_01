// 参数化数据类型 - 可通过编译时选项来切换float和double
//#define USE_FLOAT
#ifdef USE_FLOAT
typedef float real_t;
#define EPSILON 1e-6f
#else
typedef double real_t;
#define EPSILON 1e-12
#endif
// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


#define L 700
// 提前全局定义 L * L
int L2;

#define ITMAX 100
#define BLOCK_SIZE 8
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define MAXEPS 0.5
// 将三维 idx 展平, 其中 L2 表示 L*L, 需要在程序中提前计算好
#define idx3D(i, j, k) ((i) * L2 + (j) * L + (k))


// 获取GPU信息的函数
void print_gpu_info() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("未找到CUDA设备\n");
        return;
    }
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

        printf("\nGPU设备 %d: \"%s\"\n", dev, deviceProp.name);
        printf("  总全局内存: %.2f GB\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f / 1024.0f);
        printf("  计算能力: %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  每块最大线程数: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  最大线程维度: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  最大网格维度: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
    // 获取可用内存
    size_t free, total;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
    printf("\n内存信息:\n");
    printf("  GPU总内存: %.2f GB\n", (float)total / 1048576.0f / 1024.0f);
    printf("  GPU可用内存: %.2f GB\n", (float)free / 1048576.0f / 1024.0f);


    int deviceId;
    cudaDeviceProp deviceProps;
    // 获取当前设备的 ID
    cudaGetDevice(&deviceId);
    // 获取设备属性
    cudaGetDeviceProperties(&deviceProps, deviceId);
    // 打印每个线程块最大共享内存大小
    printf("\n每个线程块最大共享内存大小: %d bytes\n\n", deviceProps.sharedMemPerBlock);
}















__global__ void jacobi_kernel_1_block(real_t* A, real_t* B, real_t* block_max) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int L2 = L * L;

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    extern __shared__ real_t sdata[];  // 使用共享内存存放当前线程块的数据

    //real_t tmp = 0;
    //if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
    //    int idx3d = idx3D(i, j, k);
    //    tmp = fabs(B[idx3d] - A[idx3d]); // 已计算出绝对值
    //    A[idx3d] = B[idx3d];
    //}
    //sdata[tid] = tmp; // 将绝对值存储进共享内存中

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
        int idx3d = idx3D(i, j, k);
        sdata[tid] = fabs(B[idx3d] - A[idx3d]); // 已计算出绝对值
        A[idx3d] = B[idx3d];
    }
    else
        sdata[tid] = 0;

    __syncthreads(); // 这个不能去掉

    // 二分法块内归约
    for (int s = (blockDim.x * blockDim.y * blockDim.z) >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = Max(sdata[tid], sdata[tid + s]);  // 很重要，用Max而不是max
        }
        __syncthreads(); // 必须添加(可以试试在 polus 中去掉会怎么样)
    }
    // 第 0 个线程将当前最大值写入全局内存
    if (tid == 0) {
        block_max[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = sdata[0];
    }
}

__global__ void jacobi_kernel_2(real_t* A, real_t* B) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int L2 = L * L;

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
        B[idx3D(i, j, k)] = (
                                    A[idx3D(i-1, j, k)] +
                                    A[idx3D(i, j-1, k)] +
                                    A[idx3D(i, j, k-1)] +
                                    A[idx3D(i, j, k+1)] +
                                    A[idx3D(i, j+1, k)] +
                                    A[idx3D(i+1, j, k)]
                            ) / 6.;
    }
}

//__global__ void ab(real_t* A, real_t* B) {
//    int i = blockIdx.z * blockDim.z + threadIdx.z;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    int k = blockIdx.x * blockDim.x + threadIdx.x;
//    int L2 = L * L;
//
//    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
//        int idx3d = idx3D(i, j, k);
//        A[idx3d] = B[idx3d];
//    }
//}

// ******** 初代核函数 **********************************
//__global__ void jacobi_kernel_1_global(real_t* A, real_t* B, int num_remaining) {
//    int i = blockIdx.x; // 第 i 个块
//    int tid = threadIdx.x; // 第 idx 个进程
//    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
//    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
//        // 将块内所有值存入shared mem中
//        extern __shared__ real_t sdata[];
//        sdata[tid] = A[idx_thread_global];
//        __syncthreads();
//        // 二分法块内归约
//        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
//            if (tid < s) {
//                if (tid + s + i * blockDim.x < num_remaining) {
//                    sdata[tid] = Max(sdata[tid], sdata[tid + s]);  // 很重要，用Max而不是max
//                }
//            }
//        }
//
//        //__syncthreads();
//        // 第 0 个线程将当前最大值写入全局内存
//        if (tid == 0) {
//            B[i] = sdata[0];
//        }
//    }
//}

__global__ void jacobi_kernel_1_global(real_t* A, real_t* B, int num_remaining) {
    int i = blockIdx.x; // 第 i 个块
    int tid = threadIdx.x; // 第 idx 个进程
    int idx_thread_global = tid + i * blockDim.x; // 相对于 A 数组（即 block_max）的地址位置
    // 将块内所有值存入shared mem中
    extern __shared__ real_t sdata[];
    if (idx_thread_global < num_remaining) {   // 让每个线程不超出全局线程
        sdata[tid] = A[idx_thread_global];
    }
    else {  // 对于最后一个 block 的超出部分我们填充进一个极小值
        sdata[tid] = -INFINITY;
    }
    __syncthreads();
    // 二分法块内归约
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = Max(sdata[tid], sdata[tid + s]);  // 很重要，用Max而不是max
        }
        __syncthreads();
    }

    //__syncthreads();
    // 第 0 个线程将当前最大值写入全局内存
    if (tid == 0) {
        B[i] = sdata[0];
    }
}

int jacobi_gpu(real_t* A, real_t* B, int itmax, int size, real_t maxeps) {
    int it;
    real_t eps;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE); // 设置线程块大小
    dim3 gridDim(
            (L + blockDim.x - 1) / blockDim.x,
            (L + blockDim.y - 1) / blockDim.y,
            (L + blockDim.z - 1) / blockDim.z
    );

    // 每个block的共享内存(shared mem)大小
    int thread_num_in_block = blockDim.x * blockDim.y * blockDim.z;
    int sharedMemSize = thread_num_in_block * sizeof(real_t);
    int numBlocks = gridDim.x * gridDim.y * gridDim.z;

    real_t* block_max;
    CHECK_CUDA_ERROR(cudaMalloc(&block_max, numBlocks * sizeof(real_t)));
    real_t* new_block_max;
    CHECK_CUDA_ERROR(cudaMalloc(&new_block_max, numBlocks * sizeof(real_t)));
    real_t* tmp;

    // 循环开始
    for (it = 1; it <= itmax; ++it) {
        eps = 0.;
        // 第一计算
        // 块内归约
        jacobi_kernel_1_block <<<gridDim, blockDim, sharedMemSize >>> (A, B, block_max);
        //CHECK_CUDA_ERROR(cudaDeviceSynchronize());



        // 块间归约 1
        //thrust::device_ptr<real_t> p_block_max = thrust::device_pointer_cast(block_max);
        //eps = thrust::reduce(p_block_max, p_block_max + numBlocks, 0., thrust::maximum<real_t>());

        // 块间归约 2
        //real_t* h_block_max = (real_t*)malloc(numBlocks * sizeof(real_t));
        //cudaMemcpy(h_block_max, block_max, numBlocks * sizeof(real_t), cudaMemcpyDeviceToHost);
        //for (int i = 0; i < numBlocks; ++i) {
        //    eps = Max(eps, h_block_max[i]);
        //}

        // 块间归约 3
        int num_remaining = numBlocks;
        int block_size_cube = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;  // 用于归约函数的 block 块大小

        while (num_remaining > 1) {
            int next_blocks = (num_remaining + block_size_cube - 1) / block_size_cube; // 新归约任务产生的block数

            // printf("num_remaining: %d,   block_size_cube: %d,   next_blocks: %d\n", num_remaining, block_size_cube, next_blocks);

            jacobi_kernel_1_global <<<next_blocks, block_size_cube, block_size_cube * sizeof(real_t) >>> (block_max, new_block_max, num_remaining); // 将结果存储在 new_block_max 中

            // 交换 block_max 和 new_block_max 值（交换指针）
            tmp = block_max;
            block_max = new_block_max;
            new_block_max = tmp;

            num_remaining = next_blocks;
        }
        // Copy final result to host
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, block_max, sizeof(real_t), cudaMemcpyDeviceToHost));
        // 块间归约 3 end


        // ab << < gridDim, blockDim >> > (A, B);
        // 第二计算
        jacobi_kernel_2 <<<gridDim, blockDim >>> (A, B);

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }
    CHECK_CUDA_ERROR(cudaFree(block_max));
    CHECK_CUDA_ERROR(cudaFree(new_block_max));
    return it;
}

























//###############################################################################################################
//###############################################################################################################
int main(int argc, char** argv) {
    int i, j, k;
    double startt, endt, cpu_time;
    int iterations;

    L2 = L * L;

    print_gpu_info();
    // 检查问题大小是否适合GPU内存
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_mem = 2 * L * L * L * sizeof(real_t); // A和B数组
    if (required_mem > free_mem) {
        printf("错误: 问题大小对于GPU内存太大。\n");
        printf("所需内存: %.2f GB, 可用: %.2f GB, 总共内存: %.2f GB\n",
               (float)required_mem / 1048576.0f / 1024.0f,
               (float)free_mem / 1048576.0f / 1024.0f,
               (float)total_mem / 1048576.0f / 1024.0f);
        return 1;
    }

    // 分配cpu数组
    real_t* A_data = (real_t*)calloc(L * L * L, sizeof(real_t));
    real_t* B_data = (real_t*)calloc(L * L * L, sizeof(real_t));
    if (A_data == NULL || B_data == NULL) { printf("内存分配失败！\n"); return 1; }
    // 初始化数组
    for (i = 0; i < L; ++i)
        for (j = 0; j < L; ++j)
            for (k = 0; k < L; ++k) {
                //A_data[idx3D(i, j, k)] = 0; // 不需要，由于使用 calloc，因此数组生成时值已经是 0
                if ( !(i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) )
                    B_data[idx3D(i, j, k)] = 4 + i + j + k;
            }
    // GPU 副本
    real_t* d_A = NULL;
    real_t* d_B = NULL;
    // 分配内存
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, L * L * L * sizeof(real_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, L * L * L * sizeof(real_t)));
    // 赋值
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));

    printf("GPU计算计时开始\n");
    startt = omp_get_wtime();
    iterations = jacobi_gpu(d_A, d_B, ITMAX, L, MAXEPS);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 等待GPU任务运行完毕
    endt = omp_get_wtime();

    printf("\n Jacobi3D HZRL GPU基准测试已完成。\n");
    printf(" 大小            = %4d x %4d x %4d\n", L, L, L);
    printf(" 迭代次数        = %d\n", --iterations);
    printf(" 运行时间（秒）  = %.6f\n", endt - startt);
    printf(" 操作类型        = %s精度\n", sizeof(real_t) == sizeof(double) ? "双" : "单");

    // 释放内存
    free(A_data);
    free(B_data);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));

    printf("END OF Jacobi3D Benchmark\n");

    return 0;
}
