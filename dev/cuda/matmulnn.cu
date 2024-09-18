#include <cuda_runtime.h>
#include <stdio.h>
#include <string>
#include <chrono>
#include "common.h"

template <class DT = std::chrono::milliseconds,
          class ClockT = std::chrono::steady_clock>
class Timer {
  using timep_t = typename ClockT::time_point;
  timep_t _start = ClockT::now(), _end = {};

 public:
  void tick() {
    _end = timep_t{};
    _start = ClockT::now();
  }

  void tock() { _end = ClockT::now(); }

  template <class T = DT>
  auto duration() const {
    return std::chrono::duration_cast<T>(_end - _start);
  }
};

void matmulnn_cpu(float *A, float *B, float *C, int N) {
  for (int row = 0; row < N; ++row) {
    for (int col = 0; col < N; ++col) {
      float sum = 0.f;
      for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
      }
      C[row * N + col] = sum;
    }
  }
}

__global__ void matmulnn_kernel(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float sum = 0.f;
    for (int k = 0; k < N; ++k) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

#define TILE_MAX_DIM 32

__global__ void matmulnn_kernel_tiling(float *A, float *B, float *C, int N, int tile_dim = TILE_MAX_DIM) {
  __shared__ float A_s[TILE_MAX_DIM][TILE_MAX_DIM];
  __shared__ float B_s[TILE_MAX_DIM][TILE_MAX_DIM];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  float sum = 0.f;
  
  for (int tile = 0; tile < N / tile_dim; ++tile) {
    A_s[threadIdx.y][threadIdx.x] = A[row * N + tile * tile_dim + threadIdx.x];
    B_s[threadIdx.y][threadIdx.x] = B[(tile * tile_dim + threadIdx.y) * N + col];
    __syncthreads();
    for (int k = 0; k < tile_dim; ++k) {
      sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
    }
    __syncthreads();
  }

  C[row * N + col] = sum;

}

void matmulnn_gpu(float *A, float *B, float *C, unsigned int N, bool tiling, int blockdim = 32) {
  Timer<std::chrono::microseconds> timer;

  timer.tick();
  // Allocate memory on GPU
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, N * N * sizeof(float));
  cudaMalloc((void **)&B_d, N * N * sizeof(float));
  cudaMalloc((void **)&C_d, N * N * sizeof(float));
  cudaDeviceSynchronize();

  timer.tock();
  printf("Allocate time: %ld us\n", timer.duration().count());

  // Copy data from CPU to GPU
  timer.tick();

  cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  timer.tock();

  printf("Copy to GPU time: %ld us\n", timer.duration().count());

  // Call kernel
  timer.tick();
  dim3 threadsPerBlock(blockdim, blockdim);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  if (!tiling) {
    matmulnn_kernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);
  } else {
    matmulnn_kernel_tiling<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N, blockdim);
  }
  
  cudaDeviceSynchronize();
  timer.tock();
  printf("Kernel time: %ld us\n", timer.duration().count());

  // Copy data from GPU to CPU
  timer.tick();
  cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  timer.tock();
  printf("Copy from GPU time: %ld us\n", timer.duration().count());

  // Free memory
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main(int argc, char const *argv[]) {
  bool tiling = false;
  int N = 1024;
  int blockdim = 32;
  if (argc > 1) {
    if (argv[1] == std::string("help")) {
      printf("Usage: ./matmul <N> <blockdim> [tiling]\n");
      return 0;
    }
    N = atoi(argv[1]);
    if (argc > 2) {
      blockdim = atoi(argv[2]);
    }
    
    if (argc > 3) {
      tiling = argv[3] == std::string("tiling") ? true : false;
    }
  }
  
  float *A = make_random_float(N * N);
  float *B = make_random_float(N * N);
  float *C = (float *)malloc(N * N * sizeof(float));

  Timer<std::chrono::milliseconds> timer;
  timer.tick();
  matmulnn_cpu(A, B, C, N);
  timer.tock();

  printf("cpu time: %ld ms\n", timer.duration().count());

  float *C_2 = (float *)malloc(N * N * sizeof(float));
  timer.tick();
  matmulnn_gpu(A, B, C_2, N, false, blockdim);
  timer.tock();
  printf("gpu time: %ld ms\n", timer.duration().count());

  // check result
  int nfaults = 0;
  float tolerance = 1e-4;
  float epsilon = FLT_EPSILON;
    for (int i = 0; i < N * N; i++) {
      // Skip masked elements
      if(!isfinite(C[i]))
          continue;

      // print the first few comparisons
      if (i < 5) {
          printf("%f %f\n", C[i], C_2[i]);
      }
      // effective tolerance is based on expected rounding error (epsilon),
      // plus any specified additional tolerance
      float t_eff = tolerance + fabs(C[i]) * epsilon;
      // ensure correctness for all elements.
      if (fabs(C[i] - C_2[i]) > t_eff) {
          printf("Mismatch at %d: CPU_ref: %f vs GPU: %f\n", i, C[i], C_2[i]);
          nfaults ++;
          if (nfaults >= 10) {
              exit(EXIT_FAILURE);
          }
      }
  }

  if (tiling) {
    printf("Tiling\n");
    float *C_3 = (float *)malloc(N * N * sizeof(float));
    timer.tick();
    matmulnn_gpu(A, B, C_3, N, true, blockdim);
    timer.tock();
    printf("gpu time: %ld ms\n", timer.duration().count());

    // check result
    int nfaults = 0;
    float tolerance = 1e-4;
    float epsilon = FLT_EPSILON;
    for (int i = 0; i < N * N; i++) {
      // Skip masked elements
      if (!isfinite(C[i])) continue;

      // print the first few comparisons
      if (i < 5) {
        printf("%f %f\n", C[i], C_3[i]);
      }
      // effective tolerance is based on expected rounding error (epsilon),
      // plus any specified additional tolerance
      float t_eff = tolerance + fabs(C[i]) * epsilon;
      // ensure correctness for all elements.
      if (fabs(C[i] - C_3[i]) > t_eff) {
        printf("Mismatch at %d: CPU_ref: %f vs GPU: %f\n", i, C[i], C_3[i]);
        nfaults++;
        if (nfaults >= 10) {
          exit(EXIT_FAILURE);
        }
      }
    }
    free(C_3);
  }

  free(A);
  free(B);
  free(C);
  free(C_2);
  printf("Done!\n");
  return 0;
}
