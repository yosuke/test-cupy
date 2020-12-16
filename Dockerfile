FROM ros:melodic

ENV CUDA_VERSION 10.1.105
ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
ENV NCCL_VERSION 2.4.8
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV CUDNN_VERSION 7.6.5.32

RUN apt-get update && apt-get install -y curl

RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-1 \
&& ln -s cuda-10.1 /usr/local/cuda

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
cuda-libraries-$CUDA_PKG_VERSION \
cuda-npp-$CUDA_PKG_VERSION \
cuda-nvtx-$CUDA_PKG_VERSION \
libcublas10=10.2.1.243-1 \
libnccl2=$NCCL_VERSION-1+cuda10.1 \
&& apt-mark hold libnccl2

RUN apt-mark hold libcublas10

RUN apt-get update && apt-get install -y --no-install-recommends \
cuda-nvml-dev-$CUDA_PKG_VERSION \
cuda-command-line-tools-$CUDA_PKG_VERSION \
cuda-nvprof-$CUDA_PKG_VERSION \
cuda-npp-dev-$CUDA_PKG_VERSION \
cuda-libraries-dev-$CUDA_PKG_VERSION \
cuda-minimal-build-$CUDA_PKG_VERSION \
libcublas-dev=10.2.1.243-1 \
libnccl-dev=2.4.8-1+cuda10.1 \
&& apt-mark hold libnccl-dev

RUN apt-mark hold libcublas-dev

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends libcudnn7=$CUDNN_VERSION-1+cuda10.1 libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 && apt-mark hold libcudnn7

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python2 get-pip.py && rm get-pip.py && pip install -U pip

SHELL ["/bin/bash", "-c"]

RUN source ~/.bashrc && pip install -U cupy-cuda101==6.0.0 chainer==6.0.0 chainercv

COPY test.py /test.py

CMD python /test.py