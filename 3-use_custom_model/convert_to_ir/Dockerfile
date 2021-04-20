FROM ubuntu:18.04
WORKDIR /venv
RUN apt-get update
RUN apt -y upgrade
RUN apt -y install python3
RUN apt -y install python3-pip
RUN apt -y install cmake
RUN apt -y install protobuf-compiler libprotoc-dev
RUN python3 -m pip install "numpy>=1.12.0"
RUN python3 -m pip install "networkx>=1.11"
RUN python3 -m pip install "protobuf==3.6.1"
RUN python3 -m pip install "defusedxml>=0.5.0"
RUN python3 -m pip install "mxnet>=1.0.0,<=1.5.1"
RUN python3 -m pip install "onnx>=1.1.2"
RUN apt -y install wget
RUN wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz
RUN tar --strip-components=1 -zxvf l_openvino_toolkit_p_2020.1.023.tgz
RUN ./install_openvino_dependencies.sh