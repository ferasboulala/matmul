cmake_minimum_required(VERSION 3.0)
project(radon-tf LANGUAGES CXX)

set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ALL_LIBRARIES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_compile_options(
  -Ofast
  -march=native
  -Wall
  -Werror
  -std=c++2a
  -std=gnu++2a
  -fverbose-asm
  -save-temps
)

add_subdirectory(benchmarks)