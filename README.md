# PyTorch Primitives from Scratch 🛠️

This repository contains manual implementations of core Deep Learning layers (`BatchNorm`, `Dropout`) using PyTorch. 

The goal of this project is to deconstruct the "black box" of standard libraries and demonstrate a deep understanding of them

## 📂 Implemented Modules
### 1. Batch Normalization (`MyBatchNorm1d`, `MyBatchNorm2d`)
A complete implementation of Batch Normalization as described in the [original paper](https://arxiv.org/abs/1502.03167).

**Key Engineering Challenges Solved:**
* **Variance Estimators:** correctly distinguishing between **biased variance** (population variance formula, used for normalizing the current batch) and **unbiased variance** (Bessel's correction, used for updating global running statistics).
* **Inference Mode:** Implementing the switch between batch statistics (training) and running statistics (inference) using `register_buffer`.
* **Gradient Management:** Using `torch.no_grad()` to update running mean/var without retaining the computation graph (preventing memory leaks).
* **Broadcasting:** Handling proper tensor reshaping `(1, C, 1, 1)` to apply channel-wise normalization on 4D tensors (NCHW).

### 2. Dropout (`MyDropout`)
Implementation of **Inverted Dropout**.

**Key Concepts:**
* **Scaling:** Scaling the active neurons by `1/(1-p)` during training. This ensures that the expected value of the activation remains constant, removing the need to scale weights during inference.
* **Masking:** efficient generation of the binary mask using Bernoulli distribution.
