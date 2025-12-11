# BroTorch: Deep Learning Framework from Scratch 🚀

**My own Deep Learning Library** is a lightweight, modular deep learning library built entirely from scratch using only **NumPy**. 

The main goal of this project is to demystify the "black box" nature of modern frameworks like PyTorch and TensorFlow. It implements the core mathematical concepts of deep learning—Forward Propagation, Backpropagation, Chain Rule, and Gradient Descent—without relying on automatic differentiation engines.

Using the philosophy of **"You don't understand it until you build it,"** I aimed to improve my understanding of deep learning.

## 🌟 Features

* **Pure Math:** No Autograd. Every derivative is calculated manually using Matrix Calculus.
* **Modular Design:** Layers, Optimizers, and Loss functions are decoupled and reusable.
* **Layers:**
    * `Dense` (Fully Connected) with **He Initialization**.
    * `ReLU` Activation.
    * `Softmax` Activation (Stable version).
* **Loss Function:** `Categorical Cross-Entropy`.
* **Optimization:** `Stochastic Gradient Descent (SGD)` with support for **L2 Regularization (Weight Decay)**.
