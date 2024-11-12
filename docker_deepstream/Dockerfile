FROM nvcr.io/nvidia/deepstream:6.0-samples

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update -y 

RUN apt install -y mc htop nano git wget zip python3-pip python3-gst-1.0

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps.git

RUN cd deepstream_reference_apps && git checkout DS_6.0

RUN cd deepstream_reference_apps/deepstream_app_tao_configs/ && cp -a * /opt/nvidia/deepstream/deepstream-6.0/samples/configs/tao_pretrained_models/

RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.0/pyds-1.1.0-py3-none-linux_x86_64.whl

RUN pip3 install pyds-1.1.0-py3-none-linux_x86_64.whl

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

RUN cd deepstream_python_apps && git checkout v1.1.0

RUN cd deepstream_python_apps && mkdir /opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/ && cp -a *  /opt/nvidia/deepstream/deepstream-6.0/sources/deepstream_python_apps/

RUN pip3 install --upgrade pip

RUN pip3 install opencv-python

#RUN cd /opt/nvidia/deepstream/deepstream-6.0/samples/configs/tao_pretrained_models/ && ./download_models.sh
