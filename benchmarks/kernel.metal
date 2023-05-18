#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void matmul_tiled(
    device const T* A,
    device const T* B,
    device T* C,
    device uint32_t &N,
    device uint32_t &K,
    device uint32_t &M,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]) {
    constexpr uint64_t TILE_SIZE = 8;

    threadgroup T tileA[TILE_SIZE][TILE_SIZE];
    threadgroup T tileB[TILE_SIZE][TILE_SIZE];
    T c = T(0);

    const uint i = gid[0] * TILE_SIZE + tid[0];
    const uint j = gid[1] * TILE_SIZE + tid[1];

    for (uint outer = 0; outer < K / TILE_SIZE; ++outer) {
        const uint ii = outer * TILE_SIZE + tid[0];
        const uint jj = outer * TILE_SIZE + tid[1];

        tileA[tid[0]][tid[1]] = A[i * K + jj];
        tileB[tid[0]][tid[1]] = B[ii * M + j];

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint inner = 0; inner < TILE_SIZE; ++inner) {
            c += tileA[tid[0]][inner] * tileB[inner][tid[1]];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    C[i * M + j] = c;

}

template [[host_name("matmul_tiled_fp32")]]
kernel void matmul_tiled<float>(
    device const float* A,
    device const float* B,
    device float* C,
    device uint32_t &N,
    device uint32_t &K,
    device uint32_t &M,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]);

template [[host_name("matmul_tiled_int32")]]
kernel void matmul_tiled<int32_t>(
    device const int32_t* A,
    device const int32_t* B,
    device int32_t* C,
    device uint32_t &N,
    device uint32_t &K,
    device uint32_t &M,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]);

template [[host_name("matmul_tiled_int64")]]
kernel void matmul_tiled<int64_t>(
    device const int64_t* A,
    device const int64_t* B,
    device int64_t* C,
    device uint32_t &N,
    device uint32_t &K,
    device uint32_t &M,
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]);