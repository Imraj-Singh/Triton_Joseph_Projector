FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
#pytorch:2.3.0-cuda12.1-cudnn8-devel
RUN apt-get update

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal dependencies: git for source control, wget to download Miniconda
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# pip install -U "jax[cuda12]"
# pip install absl-py
# RUN conda install -c conda-forge parallelproj cupy notebook ipywidgets -y

#RUN python3 -m pip install --no-cache-dir --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#RUN conda install -c conda-forge parallelproj pytorch cupy notebook ipwidgets -y
# Verify installation (optional)
#RUN echo "Verifying JAX GPU detection..." && \
#    python -c "import jax; print(f'JAX Found Devices: {jax.devices()}')"