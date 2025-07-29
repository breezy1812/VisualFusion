rm -rf build && mkdir build && cd build && cmake -DCMAKE_PREFIX_PATH=/opt/libtorch .. && make -j$(nproc) 

