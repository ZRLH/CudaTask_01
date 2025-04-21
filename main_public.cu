// USE_FLOAT --> real_t will be floated, but double
//#define USE_FLOAT
#ifdef USE_FLOAT
typedef float real_t;
#define EPSILON 1e-6f
#else
typedef double real_t;
#define EPSILON 1e-12
#endif
// CUDA error macro checker
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>


#define L 900
#define ITMAX 20
// global L
int L2; // L2 : L*L

#define BLOCK_SIZE 8
#define Max(a, b) ((a) > (b) ? (a) : (b))
#define MAXEPS 0.5
// L2 : L*L
#define idx3D(i, j, k) ((i) * L2 + (j) * L + (k))


// get GPU info
void print_gpu_info() {
    int deviceCount;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("Not found CUDA device\n");
        return;
    }
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

        printf("\nGPU Device %d: \"%s\"\n", dev, deviceProp.name);
        printf("  Total Global Memory: %.2f GB\n",
               (float)deviceProp.totalGlobalMem / 1048576.0f / 1024.0f);
        printf("  Compute Capability: %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Thread Dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
    // Get available memory
    size_t free, total;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
    printf("\nMemory Info:\n");
    printf("  Total GPU Memory: %.2f GB\n", (float)total / 1048576.0f / 1024.0f);
    printf("  Available GPU Memory: %.2f GB\n", (float)free / 1048576.0f / 1024.0f);


    int deviceId;
    cudaDeviceProp deviceProps;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProps, deviceId);
    printf("\nMaximum shared memory size per thread block: %d bytes\n\n", deviceProps.sharedMemPerBlock);
}



__global__ void jacobi_kernel_1_block(real_t* A, real_t* B, real_t* block_max) {
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int L2 = L * L;

    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    extern __shared__ real_t sdata[];

    if (i > 0 && i < L - 1 && j > 0 && j < L - 1 && k > 0 && k < L - 1) {
        int idx3d = idx3D(i, j, k);
        sdata[tid] = fabs(B[idx3d] - A[idx3d]); // The absolute value has been calculated
        A[idx3d] = B[idx3d];
    }
    else
        sdata[tid] = 0;

    __syncthreads();

    // Binary block reduction
    for (int s = (blockDim.x * blockDim.y * blockDim.z) >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = Max(sdata[tid], sdata[tid + s]);  // use Max, but max
        }
        __syncthreads(); // 必须添加(可以试试在 polus 中去掉会怎么样)
    }
    // Thread 0 writes the current maximum value to global memory
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

__global__ void jacobi_kernel_1_global(real_t* A, real_t* B, int num_remaining) {
    int i = blockIdx.x; // i-th block
    int tid = threadIdx.x; // process idx
    int idx_thread_global = tid + i * blockDim.x; // The address location relative to the A array (i.e. block_max)
    // Store all values in the block into shared mem
    extern __shared__ real_t sdata[];
    if (idx_thread_global < num_remaining) {   // Keep each thread within the global thread
        sdata[tid] = A[idx_thread_global];
    }
    else {  // For the excess part of the last block, we fill it with a minimum value
        sdata[tid] = -INFINITY;
    }
    __syncthreads();
    // Binary block reduction
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = Max(sdata[tid], sdata[tid + s]);  // use Max, but max
        }
        __syncthreads();
    }

    //__syncthreads();
    // Thread 0 writes the current maximum value to global memory
    if (tid == 0) {
        B[i] = sdata[0];
    }
}

real_t jacobi_gpu(real_t* A, real_t* B, int itmax, int size, real_t maxeps) {
    int it;
    real_t eps;

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE); // Set thread block size
    dim3 gridDim(
            (L + blockDim.x - 1) / blockDim.x,
            (L + blockDim.y - 1) / blockDim.y,
            (L + blockDim.z - 1) / blockDim.z
    );

    // Shared memory size of each block
    int thread_num_in_block = blockDim.x * blockDim.y * blockDim.z;
    int sharedMemSize = thread_num_in_block * sizeof(real_t);
    int numBlocks = gridDim.x * gridDim.y * gridDim.z;

    real_t* block_max;
    CHECK_CUDA_ERROR(cudaMalloc(&block_max, numBlocks * sizeof(real_t)));
    real_t* new_block_max;
    CHECK_CUDA_ERROR(cudaMalloc(&new_block_max, numBlocks * sizeof(real_t)));
    real_t* tmp;

    // Loop Start
    for (it = 1; it <= itmax; ++it) {
        eps = 0.;
        // First Calculation
        // In-block reduction
        jacobi_kernel_1_block <<<gridDim, blockDim, sharedMemSize >>> (A, B, block_max);


        // Inter-block reduction 1
        //thrust::device_ptr<real_t> p_block_max = thrust::device_pointer_cast(block_max);
        //eps = thrust::reduce(p_block_max, p_block_max + numBlocks, 0., thrust::maximum<real_t>());

        // Inter-block reduction 2
        //real_t* h_block_max = (real_t*)malloc(numBlocks * sizeof(real_t));
        //cudaMemcpy(h_block_max, block_max, numBlocks * sizeof(real_t), cudaMemcpyDeviceToHost);
        //for (int i = 0; i < numBlocks; ++i) {
        //    eps = Max(eps, h_block_max[i]);
        //}

        // Inter-block reduction 3
        int num_remaining = numBlocks;
        int block_size_cube = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;  // The block size used for the reduction function

        while (num_remaining > 1) {
            int next_blocks = (num_remaining + block_size_cube - 1) / block_size_cube; // The number of blocks generated by the new reduction task

            jacobi_kernel_1_global <<<next_blocks, block_size_cube, block_size_cube * sizeof(real_t) >>> (block_max, new_block_max, num_remaining); // Store the result in new_block_max

            // Swap block_max and new_block_max values (swap pointers!!)
            tmp = block_max;
            block_max = new_block_max;
            new_block_max = tmp;

            num_remaining = next_blocks;
        }
        // Copy final result to host
        CHECK_CUDA_ERROR(cudaMemcpy(&eps, block_max, sizeof(real_t), cudaMemcpyDeviceToHost));
        // Inter-block reduction 3 end


        // Second calculation
        jacobi_kernel_2 <<<gridDim, blockDim >>> (A, B);

        printf(" IT = %4i   EPS = %14.7E\n", it, eps);
        if (eps < MAXEPS)
            break;
    }
    CHECK_CUDA_ERROR(cudaFree(block_max));
    CHECK_CUDA_ERROR(cudaFree(new_block_max));
    return eps;
}

























//###############################################################################################################
//###############################################################################################################
int main(int argc, char** argv) {
    int i, j, k;
    double startt, endt;
    real_t eps;

    L2 = L * L;

    print_gpu_info();
    // Check if the problem size fits in GPU memory
    size_t free_mem, total_mem;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free_mem, &total_mem));
    size_t required_mem = 2 * L * L * L * sizeof(real_t); // A and B arrays
    if (required_mem > free_mem) {
        printf("Error: Matrix size too large for GPU memory.\n");
        printf("Required memory: %.2f GB, Available: %.2f GB, Total memory: %.2f GB\n",
               (float)required_mem / 1048576.0f / 1024.0f,
               (float)free_mem / 1048576.0f / 1024.0f,
               (float)total_mem / 1048576.0f / 1024.0f);
        return 1;
    }

    // Allocate cpu array
    real_t* A_data = (real_t*)calloc(L * L * L, sizeof(real_t));
    real_t* B_data = (real_t*)calloc(L * L * L, sizeof(real_t));
    if (A_data == NULL || B_data == NULL) { printf("Memory allocation failed！\n"); return 1; }
    // Initializing an Array
    for (i = 0; i < L; ++i)
        for (j = 0; j < L; ++j)
            for (k = 0; k < L; ++k) {
                //A_data[idx3D(i, j, k)] = 0; // No need, cause use calloc, the array is already initialized to 0 upon creation.
                if ( !(i == 0 || j == 0 || k == 0 || i == L - 1 || j == L - 1 || k == L - 1) )
                    B_data[idx3D(i, j, k)] = 4 + i + j + k;
            }
    // GPU copy
    real_t* d_A = NULL;
    real_t* d_B = NULL;
    // Allocating Memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, L * L * L * sizeof(real_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, L * L * L * sizeof(real_t)));
    // Assignment
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B_data, L * L * L * sizeof(real_t), cudaMemcpyHostToDevice));

    printf("GPU calculation timing starts\n");
    startt = omp_get_wtime();
    eps = jacobi_gpu(d_A, d_B, ITMAX, L, MAXEPS);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    endt = omp_get_wtime();

    printf("\n Jacobi3D Benchmark Completed.\n");
    printf(" Size            = %4d x %4d x %4d\n", L, L, L);
    printf(" Iterations      = %d\n", ITMAX);
    printf(" Time in seconds = %.6f\n", endt - startt);
    printf(" Operation type  = %s\n", sizeof(real_t) == sizeof(double) ? "double" : "float");
    printf(" Verification    =       %12s\n", (fabs(eps - 5.058044) < 1e-11 ? "SUCCESSFUL" : "UNSUCCESSFUL"));

    // free memory
    free(A_data);
    free(B_data);
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));

    printf("END OF Jacobi3D Benchmark\n");

    return 0;
}
