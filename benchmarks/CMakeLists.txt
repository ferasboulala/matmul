# CPU
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)
add_executable(cpu-bench cpu.cpp)
target_compile_options(cpu-bench PRIVATE "-fopenmp=libomp")
target_link_libraries(cpu-bench -pthread ${OpenCV_LIBS} benchmark::benchmark /opt/homebrew/opt/libomp/lib/libomp.a)

# GPU
find_library(METAL Metal)
find_library(FOUNDATION Foundation)
find_library(METAL_KIT MetalKit)

function(build_metal_compute_shader output_file entry_point)
  set(BUILD_KERNEL_CMD   "xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${entry_point} -o ${CMAKE_BINARY_DIR}/kernel -O3")
  set(BUNDLE_LIBRARY_CMD "xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/kernel                 -o ${output_file}")
  add_custom_command(
    COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${entry_point} -o ${CMAKE_BINARY_DIR}/${entry_point}_kernel
    COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/${entry_point}_kernel  -o ${output_file}
    OUTPUT "${output_file}"
    DEPENDS ${entry_point}
    VERBATIM)
endfunction()

add_executable(gpu-bench gpu.cpp)
target_include_directories(
  gpu-bench
  PUBLIC
  ${CMAKE_SOURCE_DIR}/metal-cpp)
build_metal_compute_shader(
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu-bench-library"
  kernel.metal)
add_custom_target(
  "gpu-bench-compute-shader"
  DEPENDS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu-bench-library")
add_dependencies(gpu-bench "gpu-bench-compute-shader")
target_link_libraries(gpu-bench benchmark::benchmark ${METAL} ${FOUNDATION} ${METAL_KIT})