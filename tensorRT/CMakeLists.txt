cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(out)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(OpenCV_DIR "/usr/include/opencv4")

# Add ONNX Runtime support
set(ONNXRUNTIME_ROOT_PATH "/circ330/onnxruntime-linux-x64-1.20.1")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")
list(APPEND CMAKE_LIBRARY_PATH "${ONNXRUNTIME_ROOT_PATH}/lib")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui calib3d videoio dnn)

# TensorRT (手動指定路徑)
set(TENSORRT_INCLUDE_DIR /opt/TensorRT-8.6.1.6/include)
set(TENSORRT_LIB /opt/TensorRT-8.6.1.6/lib/libnvinfer.so)
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(/usr/local/cuda/include)
link_directories(/opt/TensorRT-8.6.1.6/lib)
link_directories(/usr/local/cuda/lib64)

link_directories(${OpenCV_LIBS} /usr/local/lib)
link_directories(${PROJECT_SOURCE_DIR}/utils/src)
link_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/src)
link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/nlohmann)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
include_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/include)
include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

add_executable(out main.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_align_tensorrt.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_to_gray.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_resizer.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_fusion.cpp
    ${PROJECT_SOURCE_DIR}/lib_image_fusion/src/core_image_perspective.cpp
    ${PROJECT_SOURCE_DIR}/utils/src/util_timer.cpp
)
target_link_libraries(out
    ${OpenCV_LIBS}
    ${TENSORRT_LIB}
    nvinfer
    nvparsers
    nvonnxparser
    cudart
    ${CUDA_LIBRARIES}
)

