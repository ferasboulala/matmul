#include <benchmark/benchmark.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <vector>

#include "common.hpp"

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"

template <typename Func, typename... Args, class TimeDuration = std::chrono::nanoseconds>
static auto measureTime(Func func, Args &&...args) {
    using FuncReturnType = decltype(func(std::forward<Args>(args)...));

    const auto start = std::chrono::high_resolution_clock::now();
    if constexpr (std::is_same_v<void, FuncReturnType>) {
        func(std::forward<Args>(args)...);
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<TimeDuration>(end - start).count();
    } else {
        FuncReturnType ret = func(std::forward<Args>(args)...);
        auto           end = std::chrono::high_resolution_clock::now();
        return std::tuple<FuncReturnType, uint64_t>{
            ret, std::chrono::duration_cast<TimeDuration>(end - start).count()};
    }
}

template <typename T>
void benchmark_matmul(benchmark::State &state) {
    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    assert(device);

    const char   *libraryPath = "bin/gpu-bench-library";
    NS::Error    *error = nullptr;
    MTL::Library *computeShaderLibrary =
        device->newLibrary(NS::String::string(libraryPath, NS::ASCIIStringEncoding), &error);
    assert(computeShaderLibrary);

    static const char *kernelName;
    if constexpr (std::is_same<T, float>::value) {
        kernelName = "matmul_tiled_fp32";
    } else if constexpr (std::is_same<T, int32_t>::value) {
        kernelName = "matmul_tiled_int32";
    } else if constexpr (std::is_same<T, int64_t>::value) {
        kernelName = "matmul_tiled_int64";
    } else {
        __builtin_unreachable();
    }

    const auto     kernelNameNS = NS::String::string(kernelName, NS::ASCIIStringEncoding);
    MTL::Function *kernelHandle = computeShaderLibrary->newFunction(kernelNameNS);
    assert(kernelHandle);

    auto computePipelineState = device->newComputePipelineState(kernelHandle, &error);
    assert(computePipelineState);

    auto commandQueue = device->newCommandQueue();
    assert(commandQueue);

    const uint32_t N = state.range(0);
    const uint64_t nBytes = N * N * sizeof(T);
    auto           A = device->newBuffer(nBytes, MTL::ResourceStorageModeShared);
    auto           B = device->newBuffer(nBytes, MTL::ResourceStorageModeShared);
    auto           C = device->newBuffer(nBytes, MTL::ResourceStorageModeShared);

    std::iota((T *)A->contents(), (T *)((char *)A->contents() + nBytes), T(0));
    std::iota((T *)B->contents(), (T *)((char *)B->contents() + nBytes), T(0));

    for (auto _ : state) {
        MTL::CommandBuffer *commandBuffer = commandQueue->commandBuffer();
        assert(commandBuffer);

        MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
        assert(computeEncoder);

        computeEncoder->setComputePipelineState(computePipelineState);
        computeEncoder->setBuffer(A, 0, 0);
        computeEncoder->setBuffer(B, 0, 1);
        computeEncoder->setBuffer(C, 0, 2);
        computeEncoder->setBytes(&N, sizeof(uint32_t), 3);
        computeEncoder->setBytes(&N, sizeof(uint32_t), 4);
        computeEncoder->setBytes(&N, sizeof(uint32_t), 5);

        static constexpr uint64_t TILE_SIZE = 8;

        const MTL::Size threadgroupSize = MTL::Size::Make(TILE_SIZE, TILE_SIZE, 1);
        const MTL::Size gridSize = MTL::Size::Make(N / TILE_SIZE, N / TILE_SIZE, 1);
        computeEncoder->dispatchThreadgroups(gridSize, threadgroupSize);
        computeEncoder->endEncoding();

        const auto commit = [&]() {
            commandBuffer->commit();
            commandBuffer->waitUntilCompleted();
        };
        const uint64_t durationNs = measureTime(commit);
        state.SetIterationTime(double(durationNs) / 1000000000);
    }

    state.SetItemsProcessed(2 * state.iterations() * std::pow(uint64_t(N), 3));

#ifndef NDEBUG
    if (N > 512) {
        return;
    }
    std::vector<T> D(N * N);
    matmul_naive<T>((T *)A->contents(), (T *)B->contents(), D.data(), N, N, N);
    assert(equal((T *)C->contents(), D.data(), D.size()));
#endif
}

BENCHMARK(benchmark_matmul<float>)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();
BENCHMARK_MAIN();
