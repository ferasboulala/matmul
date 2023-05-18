#include <benchmark/benchmark.h>

#include <cstdint>
#include <numeric>
#include <thread>

#include "common.hpp"

using DTYPE = float;

template <typename T>
void matmul_loop_interchange(const T *__restrict A,
                             const T *__restrict B,
                             T *__restrict C,
                             uint64_t N,
                             uint64_t M,
                             uint64_t K,
                             uint64_t nThreads = 1) {
    std::memset(C, 0, sizeof(T) * N * K);
#pragma omp parallel for num_threads(nThreads) if (nThreads > 1)
    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t j = 0; j < M; ++j) {
            for (uint64_t k = 0; k < K; ++k) {
                C[at(i, k, K)] += A[at(i, j, M)] * B[at(j, k, K)];
            }
        }
    }
}

template <typename T, uint64_t TILE_SIZE = 64>
void matmul_tiled(const T *__restrict A,
                  const T *__restrict B,
                  T *__restrict C,
                  uint64_t N,
                  uint64_t M,
                  uint64_t K,
                  uint64_t nThreads = 1) {
    if (N < TILE_SIZE || M < TILE_SIZE || K < TILE_SIZE) {
        matmul_loop_interchange(A, B, C, N, M, K, nThreads);
        return;
    }

    const auto atTiled = [&](uint64_t ii, uint64_t jj, uint64_t i, uint64_t j, uint64_t cols) {
        return (ii + i) * cols + jj + j;
    };

    std::memset(C, 0, sizeof(T) * N * K);

#pragma omp parallel for num_threads(nThreads) if (nThreads > 1)
    for (uint64_t ii = 0; ii < N; ii += TILE_SIZE) {
        for (uint64_t jj = 0; jj < M; jj += TILE_SIZE) {
            for (uint64_t kk = 0; kk < K; kk += TILE_SIZE) {
                for (uint64_t i = 0; i < TILE_SIZE; ++i) {
                    for (uint64_t j = 0; j < TILE_SIZE; ++j) {
                        for (uint64_t k = 0; k < TILE_SIZE; ++k) {
                            C[atTiled(ii, kk, i, k, K)] +=
                                A[atTiled(ii, jj, i, j, M)] * B[atTiled(jj, kk, j, k, K)];
                        }
                    }
                }
            }
        }
    }
}

template <typename T, typename Func, typename... ExtraArgs>
void benchmark_matmul_inner(benchmark::State &state, Func func, ExtraArgs &&...args) {
    const uint64_t N = state.range(0);
    std::vector<T> A(N * N);
    std::vector<T> B(N * N);
    std::vector<T> C(N * N, T(0));

    std::iota(A.begin(), A.end(), 0);
    std::iota(B.begin(), B.end(), 0);

    for (auto _ : state) {
        func(A.data(), B.data(), C.data(), N, N, N, std::forward<ExtraArgs>(args)...);
    }
    state.SetItemsProcessed(2 * N * N * N * state.iterations());

#ifndef NDEBUG
    if (N > 512) {
        return;
    }
    std::vector<T> D(N * N);
    matmul_naive(A.data(), B.data(), D.data(), N, N, N);
    assert(equal(C.data(), D.data(), C.size()));
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#define benchmark_matmul(state, func, ...) \
    benchmark_matmul_inner<DTYPE>(state, func<DTYPE>, ##__VA_ARGS__)
#pragma clang diagnostic pop

void benchmark_matmul_naive(benchmark::State &state) { benchmark_matmul(state, matmul_naive); }
void benchmark_matmul_loop_interchange(benchmark::State &state) {
    benchmark_matmul(state, matmul_loop_interchange, 1);
}
void benchmark_matmul_loop_interchange_threaded(benchmark::State &state) {
    benchmark_matmul(state, matmul_loop_interchange, std::thread::hardware_concurrency());
}
void benchmark_matmul_tiled(benchmark::State &state) { benchmark_matmul(state, matmul_tiled, 1); }
void benchmark_matmul_tiled_threaded(benchmark::State &state) {
    benchmark_matmul(state, matmul_tiled, std::thread::hardware_concurrency());
}

BENCHMARK(benchmark_matmul_loop_interchange)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(benchmark_matmul_loop_interchange_threaded)->RangeMultiplier(2)->Range(64, 4096);
BENCHMARK(benchmark_matmul_tiled)->RangeMultiplier(2)->Range(64, 2048);
BENCHMARK(benchmark_matmul_tiled_threaded)->RangeMultiplier(2)->Range(64, 4096);
BENCHMARK(benchmark_matmul_naive)->RangeMultiplier(2)->Range(2, 512);
BENCHMARK_MAIN();