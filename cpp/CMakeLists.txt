cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(ysda-cpu-inference)

# set(USE_FBGEMM ON)
# set(USE_QNNPACK ON)
# set(USE_MKLDNN ON)
set(CMAKE_PREFIX_PATH pytorch-install)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")


include_directories(pytorch-install/include/)

add_executable(pytorch-exec src/benchmark.cpp)

target_link_libraries(pytorch-exec "${TORCH_LIBRARIES}")

set_property(TARGET pytorch-exec PROPERTY CXX_STANDARD 17)

