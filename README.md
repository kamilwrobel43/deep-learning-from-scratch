# PyTorch Primitives from Scratch 🛠️

This repository contains implementations of core Deep Learning layers (at this moment: `BatchNorm`, `Dropout`) using PyTorch.

The primary goal of this project is **educational**: to deconstruct the "black box" of standard libraries and gain a better understanding of the underlying mathematics, tensor operations, and PyTorch's autograd engine.

## 📂 Implemented Modules

### 1. Batch Normalization (`MyBatchNorm1d`, `MyBatchNorm2d`)
A complete implementation of Batch Normalization based on the [original paper](https://arxiv.org/abs/1502.03167).

**Implementation Details:**
* **Variance Estimation:** Handling the difference between **biased variance** (used for normalizing the current batch) and **unbiased variance** (Bessel's correction, used for tracking global running statistics).
* **Training vs. Inference:** Managing the switch between using batch statistics during training and running statistics during inference via `register_buffer`.
* **Graph Management:** Using `torch.no_grad()` to update running statistics without attaching them to the computation graph, ensuring correct memory management.

### 2. Dropout (`MyDropout`)
Implementation of the **Inverted Dropout** technique.

**Key Concepts:**
* **Scaling:** Applying the `1/(1-p)` scaling factor during training time. This keeps the expected activation value constant, removing the need to modify weights during the inference phase.
* **Masking:** Generating binary masks using the Bernoulli distribution to zero out random activations.

### 3. Layer Normalization (`MyLayerNorm`)
Implementation of the `torch.nn.LayerNorm` as described in official docs and [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)

**Details**
* Single module that can handle different shapes

**Key Concepts:**
* **Scaling:** Applying the `1/(1-p)` scaling factor during training time. This keeps the expected activation value constant, removing the need to modify weights during the inference phase.
* **Masking:** Generating binary masks using the Bernoulli distribution to zero out random activations.

## 🚀 Future Plans
I plan to regularly update this repository with new implementations as I explore more advanced Deep Learning concepts. My goal is to gradually build a comprehensive collection of neural network primitives built from scratch.
