ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++ \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN chmod +x /miniconda.sh
RUN /miniconda.sh -b -p /miniconda


RUN rm /miniconda.sh



ENV PATH=/miniconda/bin:$PATH


# Create a Python 3.6 environment


 RUN /miniconda/bin/conda install -y conda-build \
  && /miniconda/bin/conda create -y --name py36 python=3.6.7 \
  && conda create -y --name py36 python=3.6.7 \
  && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python tqdm

# Install PyTorch 1.0 Nightly
ARG CUDA
RUN conda install pytorch-nightly cudatoolkit=${CUDA} -c pytorch \
 && conda clean -ya


# Install TorchVision 

RUN conda install torchvision=0.3.0 -c pytorch

# install pycocotools
RUN git clone https://github.com/cocodataset/cocoapi \
 && cd cocoapi/PythonAPI \
 && make

# install PyTorch Detection


ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}

WORKDIR /

RUN git clone https://github.com/PeterTKovacs/zold137 \
 && cd zold137 \
 && python setup.py build develop

WORKDIR /
# full gist url available upon request  - fetch it, otherwise you get ERROR here!
RUN git clone https://gist.github.com/PeterTKovacs/b4c366c* \
 && cd b4c366c* \ 
 && cp ./giro*.txt ../zold137/datasets/giro_data/annot

# weights from the original drone project
# they must be first downloaded from the given GoogleDrive link and then pasted in the same directory as the Dockerfile

WORKDIR /zold137
COPY visdrone_model_0360000.pth  drone_demo/visdrone_model_0360000.pth

# bugs

RUN apt install nano -y
RUN apt install libgl1-mesa-glx -y

WORKDIR /
RUN git clone https://www.github.com/nvidia/apex
WORKDIR /apex
RUN python setup.py install

WORKDIR /zold137/datasets
# as default, fetches the dataset for the first video
RUN pip install pafy pandas
RUN pip install youtube_dl
RUN python data_gen.py 


