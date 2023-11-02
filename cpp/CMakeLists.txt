cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
project(ysda-cpu-inference)

set(CMAKE_PREFIX_PATH libtorch)

find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g")


include_directories(libtorch/include/)
# include_directories(aten/src)
add_executable(libtorch-exec src/libtorch.cpp)
# add_executable(linear-exec src/Linear.cpp)
# add_executable(libtorch-exec src/Linear.cpp)
# add_executable(ysda-cpu-inference src/aten.cpp)
target_link_libraries(libtorch-exec "${TORCH_LIBRARIES}")
# target_link_libraries(linear-exec "${TORCH_LIBRARIES}")
set_property(TARGET libtorch-exec PROPERTY CXX_STANDARD 17)
# set_property(TARGET linear-exec PROPERTY CXX_STANDARD 17)
