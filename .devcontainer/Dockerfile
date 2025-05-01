FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*