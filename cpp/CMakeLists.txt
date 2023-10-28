cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(ysda-cpu-inference)

set(CMAKE_PREFIX_PATH libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(libtorch/include/)
add_executable(ysda-cpu-inference src/main.cpp)
target_link_libraries(ysda-cpu-inference "${TORCH_LIBRARIES}")
set_property(TARGET ysda-cpu-inference PROPERTY CXX_STANDARD 17)  