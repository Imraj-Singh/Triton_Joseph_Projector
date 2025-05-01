# Triton Joseph Projector

This code was created during the CCP SyneRBI and CCPi Hackathon on Efficient integration of SIRF/STIR/CIL with Pytorch, held in April 2025: [https://www.ccpsynerbi.ac.uk/events/hackathon-on-pytorch/](https://www.ccpsynerbi.ac.uk/events/hackathon-on-pytorch/).

This project drew inspiration from the blog post series by Chris Lattner on democratizing AI compute, particularly the discussion around Triton: [https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls](https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls).

## Motivation

The core goal was to implement a computationally expensive projection operator used in iterative tomographic reconstruction – specifically the **Joseph forward/back-projector** – using Triton. In tomography, projection operations map between the image volume (patient anatomy) and the detector data (sinogram). The Joseph projector is one algorithm for performing this mapping, calculating the intersection length of projection rays with image voxels. Due to the large number of rays and voxels, this operation is computationally intensive but also highly parallelisable as there are weak dependencies between data elements (e.g., calculating one ray's projection is largely independent of others), making it ideal for GPU acceleration. Traditionally, high-performance GPU kernels for such tasks are written in **CUDA** for Nvidia GPUs or potentially **HIP** for AMD GPUs. While mature and capable of extracting maximum performance, CUDA creates vendor lock-in, and maintaining separate CUDA/HIP codebases increases development effort.

We aimed to explore **Triton** as an alternative that offers:
1.  **Cross-Compatibility:** The ability to write a single kernel targeting both Nvidia and AMD GPUs.
2.  **Developer Productivity:** A potentially simpler development experience using Python.

## Triton

Triton [https://triton-lang.org/main/index.html](https://triton-lang.org/main/index.html) is a modern **Embedded Domain-Specific Language (eDSL)** and compiler designed for writing high-performance GPU kernels directly within Python.

> Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.

Key aspects relevant to this project include:

* **Python Integration:** Kernels are written using Python syntax. This leverages Python's ease of use and integrates seamlessly with libraries like PyTorch, allowing kernels to operate directly on `torch.Tensor` objects residing on the GPU. This avoids the explicit memory management (`cudaMalloc`, `cudaMemcpy`, etc.) often required in C++/CUDA, reducing complexity and potential bugs.
* **JIT Compilation:** Triton kernels are compiled **Just-In-Time (JIT)** when first called. Triton's compiler analyzes the Python code and generates highly optimized machine code (e.g., PTX for Nvidia) specifically for the target GPU architecture.
* **Performance Focus:** While being a higher-level language, Triton is designed to generate code competitive with hand-written CUDA. It provides abstractions for managing GPU resources like shared memory and specifying tiling strategies, guiding the compiler to produce efficient parallel execution plans.
* **Hardware Abstraction (via MLIR):** Triton achieves its cross-vendor capability by leveraging MLIR as a compiler backend (see below). The goal is to write one Triton kernel and have the compiler generate efficient code for different GPU architectures.

## MLIR (Multi-Level Intermediate Representation)

Triton doesn't directly generate PTX or other hardware-specific code. Instead, it compiles the Python kernel code into its own Triton IR, which is then lowered to **MLIR** [https://mlir.llvm.org/](https://mlir.llvm.org/).

MLIR is a modern compiler infrastructure project (originating from the LLVM family, spearheaded by Chris Lattner) designed to address the complexities of compiling diverse software (especially in AI/ML) for diverse hardware. As Lattner envisioned in the blog post linked above:

> Could we build a unified representation that could support every AI framework, every hardware backend, and every kind of optimization—from algebraic simplification to polyhedral analysis?

**How MLIR helps Triton:**

1.  **Unified Infrastructure:** MLIR provides a common framework with multiple levels of abstraction ("dialects") to represent code. Triton uses MLIR dialects to represent the kernel's computation and parallelism.
2.  **Optimisation:** Common optimization passes can be developed within the MLIR framework and applied before generating final code for the hardware.
3.  **Hardware Targeting:** MLIR handles the complex process of "lowering" the high-level representation through various intermediate steps down to something like **LLVM IR**. LLVM IR can then be compiled into the final machine code for specific backends.

Essentially, Triton provides the productive Python front-end, while MLIR provides the powerful compiler backbone that enables optimisation and retargeting to different hardware, facilitating Triton's goal of democratising high-performance GPU programming. This project serves as a practical exploration of this toolchain for an intensive tomographic image reconstruction operation.
