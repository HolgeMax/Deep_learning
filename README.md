# Deep_learning
By Holger Max Fløe Lyng, s214776
This is a repository for 02456 deep learing course at DTU. I am a student in my master on my 9th semester.

## Lecture 1 - Neural Networks

This lecture introduced the fundamentals of **neural networks (NNs)** and **supervised learning**, forming the foundation of deep learning.

### Supervised Learning
Goal: learn a function  
\[
y = f_\phi(\mathbf{x})
\]
that maps inputs **x** to outputs **y**, using parameters **φ**.  
Training minimizes a **loss function** \(L(\phi)\), e.g. **Mean Squared Error (MSE)** for regression:
\[
\hat{\phi} = \arg\min_\phi L(\phi)
\]

We learned to compute loss for both **Gaussian regression** and **classification** (cross-entropy).

### Neural Network Architecture
A **basic NN** contains one hidden layer (K=1) with many hidden units (D).  
A **deep NN** has multiple hidden layers (K>1).

Each **neuron** computes:
\[
z = \mathbf{w}^T \mathbf{x} + b, \quad h = \sigma(z)
\]
where **σ** is an activation function such as:
- Sigmoid  
- tanh  
- **ReLU:** \( \sigma(z) = \max(0, z) \)

Networks with hidden layers are called **Multilayer Perceptrons (MLPs)** or **Feed-Forward Networks (FFNs)**.  
- **Regression output:** identity  
- **Classification output:** Softmax

### Approximation & Training
By the **Universal Approximation Theorem**, even a two-layer NN can approximate any continuous function with enough hidden units.  
Training uses **Gradient Descent** and **Backpropagation** to efficiently compute and update gradients.  
**Stochastic Gradient Descent (SGD)** generalizes this for large datasets.

**Exercises:** *Notebook 1.1 “FNN Pen and Paper.ipynb”* — implement loss, backpropagation, and SGD; see also theoretical problems (3.5, 3.11, 4.1, 4.11).

## Lecture 2 - Learning
This week introduced the core concepts for training deep learning models — **loss functions**, **optimization**, and **parameter initialization**.

### Loss Functions & MLE
Training aims to minimize a **loss function** \(L(\phi)\) that measures how far predictions \(f_\phi(x_i)\) deviate from true outputs \(y_i\).  
Using **Maximum Likelihood Estimation (MLE)**, we find parameters that maximize data likelihood — typically by minimizing the **Negative Log-Likelihood (NLL)**.

**Applications:**
- **Regression:** Assuming Gaussian noise → NLL ≈ **Mean Squared Error (MSE)**  
- **Classification:** For categorical outputs → **Cross-Entropy Loss**, using **Softmax** to map outputs to probabilities

### Optimization
Models are fit with **Gradient Descent**:
\[
\phi^{(t+1)} = \phi^{(t)} - \eta \nabla_\phi L(\phi)
\]
where \( \eta \) is the learning rate.  
To handle large datasets and avoid poor local minima, **Stochastic Gradient Descent (SGD)** uses random data batches.

**Improvements:**
- **Momentum:** Smooths updates with past gradients  
- **Adam:** Adapts learning rates via gradient moments

### Gradients & Initialization
Gradients are computed via **Backpropagation**.  
Proper initialization is essential:
- Random weights:$$
\phi_0 \sim \mathcal{N}(0, \sigma_0^2)
$$

$$
\sigma_0 = \sqrt{\frac{2}{D}}
$$
to prevent exploding/vanishing activations.

**Practical:** Implemented manual autodiff in *Notebook 2.1 — “FNN AutoDif Nanograd.ipynb”*.


## Project Idees
Multi-modal registration to mri. 
Dataset:
https://www.kaggle.com/datasets/grantmcnatt/mri-and-pet-dice-similarity-dataset?utm_source=chatgpt.com
https://www.kaggle.com/datasets/29c3607295965ebb030f2d158fec487412d84c82528dd44f8ef956aef35541aa
https://www.med.upenn.edu/cbica/brats2020/data.html?utm_source=chatgpt.com
https://sites.wustl.edu/oasisbrains/?utm_source=chatgpt.com
https://www.kaggle.com/datasets/purnimakumarrr/adhd200-preprocessed-anatomical-dataset
https://www.synapse.org/Synapse:syn64153130/wiki/
