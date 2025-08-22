cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(out)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(OpenCV_DIR "/usr/include/opencv4")

# Add ONNX Runtime support (GPU version)
set(ONNXRUNTIME_ROOT_PATH "/circ330/onnxruntime-linux-x64-gpu-1.18.0")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
list(APPEND CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")
list(APPEND CMAKE_LIBRARY_PATH "${ONNXRUNTIME_ROOT_PATH}/lib")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# Find packages
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs highgui calib3d videoio dnn)

link_directories(${OpenCV_LIBS} /usr/local/lib)
link_directories(${PROJECT_SOURCE_DIR}/utils/src)
link_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/src)
link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/nlohmann)
include_directories(${PROJECT_SOURCE_DIR}/utils/include)
include_directories(${PROJECT_SOURCE_DIR}/lib_image_fusion/include)
include_directories(${ONNXRUNTIME_ROOT_PATH}/include)

add_executable(out main.cpp)
target_link_libraries(out
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so
  -lstdc++fs
  -lpthread
)

set_property(TARGET out PROPERTY CXX_STANDARD 17)

