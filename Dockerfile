FROM amazonlinux:2

RUN yum update -y && \
    yum install -y gcc gcc-c++ make cmake3 git python3 python3-devel tar wget

# Install CMake 3.28
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.0/cmake-3.28.0-linux-x86_64.sh && \
    chmod +x cmake-3.28.0-linux-x86_64.sh && \
    ./cmake-3.28.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.28.0-linux-x86_64.sh

RUN git clone --recursive https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j4

RUN mkdir -p /opt/python/lightgbm/lib && \
    cp /LightGBM/lib_lightgbm.so /opt/python/lightgbm/lib/

CMD ["/bin/bash"]