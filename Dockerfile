FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /tmp/
