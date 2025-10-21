cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# project name
project(semla)

# avoid error when compile libtorch with opencv4.x
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# include openCV
include_directories(${OpenCV_INCLUDE_DIRS})

# include project
include_directories(${PROJECT_SOURCE_DIR}/nlohmann)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
include_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/include)

# include torch
include_directories(/home/jetson/.local/lib/python3.10/site-packages/torch/include)
include_directories(/home/jetson/.local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include)

# link opencv, lib, torch
link_directories(${OpenCV_LIBS})
# link_directories(/usr/local/lib)
# link_directories(/home/jetson/.local/lib/python3.10/site-packages/torch/lib)
link_directories(${PROJECT_SOURCE_DIR}/utils/src)
link_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/src)

# set torch
set(Boost_USE_MULTITHREAD ON)

# Set Torch paths
set(CMAKE_PREFIX_PATH /home/jetson/.local/lib/python3.10/site-packages/torch)
set(Torch_DIR /home/jetson/.local/lib/python3.10/site-packages/torch/share/cmake/Torch)

# link package
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# add needed cpp file into project
add_executable(semla main.cpp)

# link library into project
target_link_libraries(semla ${TORCH_LIBRARIES} ${OpenCV_LIBS} -lstdc++fs)

# set build properties
set_property(TARGET semla PROPERTY CXX_STANDARD 17)

# set cmake flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pthread ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# copy weight into release directory
# file(COPY
#   ${CMAKE_CURRENT_SOURCE_DIR}/SemLA_jit_cpu.zip
#   ${CMAKE_CURRENT_SOURCE_DIR}/SemLA_jit_cuda.zip
#   DESTINATION ${CMAKE_BINARY_DIR}
# )