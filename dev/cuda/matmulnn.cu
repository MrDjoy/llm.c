#include <cuda_runtime.h>
#include <stdio.h>
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

void matmulnn_gpu(float *A, float *B, float *C, unsigned int N) {
  Timer<std::chrono::milliseconds> timer;

  timer.tick();
  // Allocate memory on GPU
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **)&A_d, N * N * sizeof(float));
  cudaMalloc((void **)&B_d, N * N * sizeof(float));
  cudaMalloc((void **)&C_d, N * N * sizeof(float));
  cudaDeviceSynchronize();

  timer.tock();
  printf("Allocate time: %ld ms\n", timer.duration().count());

  // Copy data from CPU to GPU
  timer.tick();

  cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  timer.tock();

  printf("Copy to GPU time: %ld ms\n", timer.duration().count());

  // Call kernel
  timer.tick();
  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  matmulnn_kernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);
  cudaDeviceSynchronize();
  timer.tock();
  printf("Kernel time: %ld ms\n", timer.duration().count());

  // Copy data from GPU to CPU
  timer.tick();
  cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  timer.tock();
  printf("Copy from GPU time: %ld ms\n", timer.duration().count());
}

int main(int argc, char const *argv[]) {
  int N = 1024;
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
  matmulnn_gpu(A, B, C_2, N);
  timer.tock();
  printf("gpu time: %ld ms\n", timer.duration().count());

  // check result
  validate_result(C_2, C, "out", N * N, 1e-3f);

  free(A);
  free(B);
  free(C);
  free(C_2);
  printf("Done!\n");
  return 0;
}
