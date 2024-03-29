FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y git python3 python3-pip python-is-python3
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip install accelerate peft bitsandbytes transformers trl wandb

COPY . /tmp/
