cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(out)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(OpenCV_DIR "/usr/include/opencv4")

set(CMAKE_PREFIX_PATH "/opt/libtorch")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui calib3d videoio)

link_directories(${OpenCV_LIBS} /usr/local/lib)
link_directories(${PROJECT_SOURCE_DIR}/utils/src)
link_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/src)
# link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/nlohmann)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
include_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/include)


add_executable(out main.cpp)
target_link_libraries(out
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  # /usr/local/cuda/lib64/libcudart.so
  -lstdc++fs
  -lpthread
)

set_property(TARGET out PROPERTY CXX_STANDARD 17)

