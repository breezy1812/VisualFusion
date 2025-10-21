rm -rf build && mkdir build
cmake -S . -B build -DTorch_DIR="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build -j"$(nproc)"