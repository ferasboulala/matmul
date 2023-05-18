#include <array>
#include <cstdint>

#define ALWAYS_INLINE __attribute__((always_inline))

template <uint64_t RANK>
static ALWAYS_INLINE uint64_t ravel(const std::array<uint64_t, RANK> &indices,
                                    const std::array<uint64_t, RANK> &sizes) {
    uint64_t index = indices.back();
    uint64_t accumulator = sizes.back();
    for (int i = RANK - 2; i >= 0; --i) {
        index += accumulator * indices[i];
        accumulator *= sizes[i];
    }

    return index;
}

ALWAYS_INLINE uint64_t at(uint64_t i, uint64_t j, uint64_t nCols) {
    return ravel<2>({i, j}, {0, nCols});
}

template <typename T>
bool equal(const T *__restrict A, const T *__restrict B, uint64_t N) {
    static constexpr T ATOL = T(0.001);
    static constexpr T RTOL = T(0.01);
    for (uint64_t i = 0; i < N; ++i) {
        const T diff = (A[i] > B[i]) ? (A[i] - B[i]) : (B[i] - A[i]);
        if (diff > ATOL || (2 * diff / (A[i] + B[i]) > RTOL)) {
            return false;
        }
    }

    return true;
}

template <typename T>
void matmul_naive(const T *__restrict A,
                  const T *__restrict B,
                  T *__restrict C,
                  uint64_t N,
                  uint64_t M,
                  uint64_t K) {
    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t k = 0; k < K; ++k) {
            C[at(i, k, K)] = T(0);
            for (uint64_t j = 0; j < M; ++j) {
                C[at(i, k, K)] += A[at(i, j, M)] * B[at(j, k, K)];
            }
        }
    }
}