# Triton Joseph Projector

This code was created at the Hackathon: [https://www.ccpsynerbi.ac.uk/events/hackathon-on-pytorch/].

This project in-particular was inspired by the following blog post series: [https://www.modular.com/blog/democratizing-ai-compute-part-7-what-about-triton-and-python-edsls].

We were interested in writing a tensor operator (using PyTorch terminology) that is computationally expensive - i.e. Joseph Projector, in a language that allows the cross-compatibility between accelerators (Nvidia or AMD gpus).

For medical image reconstruction we often have high-dimensional data that is quite indepenent, meaning that parallel computation can greatly speed up.


## Triton

Recently there has been development of Embedded Domain-Specific Languages (eDSLs). These abstract away some of the complexities of parallel computing. An example is Triton [https://triton-lang.org/main/index.html].

> Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware.



### MLIR (Multi-Level Intermediate Representation)

As written by Chris Lattner (developer of low-level virtual machine LLVM and MLIR) in the aforementioned blog, the idea behind MLIR was:

> Could we build a unified representation that could support every AI framework, every hardware backend, and every kind of optimization—from algebraic simplification to polyhedral analysis?

