cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(ysda-cpu-inference)

# set(USE_FBGEMM ON)
# set(USE_QNNPACK ON)
# set(USE_MKLDNN ON)


#set(CMAKE_PREFIX_PATH pytorch-install)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")

set(CMAKE_PREFIX_PATH pytorch-release/pytorch-install)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_BUILD_TYPE Release)



#include_directories(pytorch-install/include/)
include_directories(pytorch-release/pytorch-install/include)
find_package(Torch REQUIRED)
add_executable(pytorch-exec src/code_generation_test.cpp)

target_link_libraries(pytorch-exec "${TORCH_LIBRARIES}")

set_property(TARGET pytorch-exec PROPERTY CXX_STANDARD 17)

