FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update && apt-get install git -y

WORKDIR /workspace
COPY . /workspace/periflow-python-sdk
RUN cd periflow-python-sdk && pip install .
