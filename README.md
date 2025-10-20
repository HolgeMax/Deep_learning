# Deep_learning
By Holger Max Fløe Lyng, s214776  
This is a repository for the 02456 Deep Learning course at DTU.  
I am a master's student on my 9th semester.

---

## Lecture 1 - Neural Networks

This lecture introduced the fundamentals of **neural networks (NNs)** and **supervised learning**, forming the foundation of deep learning.

### Supervised Learning
Goal: learn a function  
y = f_phi(x)  
that maps inputs **x** to outputs **y**, using parameters **phi**.  
Training minimizes a **loss function** L(phi), e.g. **Mean Squared Error (MSE)** for regression:  
phi_hat = argmin_phi L(phi)

We learned to compute loss for both **Gaussian regression** and **classification** (cross-entropy).

### Neural Network Architecture
A **basic NN** contains one hidden layer (K = 1) with many hidden units (D).  
A **deep NN** has multiple hidden layers (K > 1).

Each **neuron** computes:  
z = wᵀx + b  
h = σ(z)  

where **σ** is an activation function such as:
- Sigmoid  
- tanh  
- **ReLU:** σ(z) = max(0, z)

Networks with hidden layers are called **Multilayer Perceptrons (MLPs)** or **Feed-Forward Networks (FFNs)**.  
- **Regression output:** identity  
- **Classification output:** Softmax

### Approximation & Training
By the **Universal Approximation Theorem**, even a two-layer NN can approximate any continuous function with enough hidden units.  
Training uses **Gradient Descent** and **Backpropagation** to efficiently compute and update gradients.  
**Stochastic Gradient Descent (SGD)** generalizes this for large datasets.

**Exercises:** *Notebook 1.1 “FNN Pen and Paper.ipynb”* implement loss, backpropagation, and SGD; see also theoretical problems (3.5, 3.11, 4.1, 4.11).

---

## Lecture 2 - Learning

This week introduced the core concepts for training deep learning models, **loss functions**, **optimization**, and **parameter initialization**.

### Loss Functions & MLE
Training aims to minimize a **loss function** L(phi) that measures how far predictions f_phi(x_i) deviate from true outputs y_i.  
Using **Maximum Likelihood Estimation (MLE)**, we find parameters that maximize data likelihood, typically by minimizing the **Negative Log-Likelihood (NLL)**.

**Applications:**
- **Regression:** Assuming Gaussian noise → NLL ≈ **Mean Squared Error (MSE)**  
- **Classification:** For categorical outputs → **Cross-Entropy Loss**, using **Softmax** to map outputs to probabilities

### Optimization
Models are fit with **Gradient Descent**:  
phi(t+1) = phi(t) - eta * ∇_phi L(phi)  
where eta is the learning rate.  

To handle large datasets and avoid poor local minima, **Stochastic Gradient Descent (SGD)** uses random data batches.

**Improvements:**
- **Momentum:** Smooths updates with past gradients  
- **Adam:** Adapts learning rates via gradient moments

### Gradients & Initialization
Gradients are computed via **Backpropagation**.  
Proper initialization is essential:
- Random weights: phi_0 ~ N(0, σ0²) 
- **He Initialization** for ReLU: σ0 = √(2/D)

to prevent exploding/vanishing activations.

**Practical:** Implemented manual autodiff in *Notebook 2.1 “FNN AutoDif Nanograd.ipynb”*.

## Week 3: Tricks of the Trade

This week covered key methods for improving **performance**, **stability**, and **generalization** in deep learnin, including performance metrics, regularization, and **residual networks**.

---

### Measuring Performance & Bias–Variance Trade-off
Goal: understand and reduce **generalization error** (test performance).  
Test error sources: **Noise**, **Bias**, and **Variance**.

- **Bias–Variance Trade-off:** Higher model complexity → lower bias but higher variance (risk of overfitting).  
- **Data Splitting:**  
  - D_train - parameter fitting  
  - D_val - hyperparameter tuning  
  - D_test - final evaluation  
- **Double Descent:** Test error can rise and fall again as capacity increases past perfect training fit (over-parameterized regime).

---

### Regularization
Used to prevent overfitting and improve generalization.

- **Explicit Regularization:** Adds penalty term g(phi) to the loss L(phi).  
  Example: L2-regularization (Weight Decay) penalizes large weights.
- **Implicit Regularization:** Arises from the optimization process itself, e.g., SGD tends to favor smoother, stable solutions.
- **Heuristics:**  
  - *Early Stopping:* stop when validation error increases  
  - *Ensembling:* average multiple models  
  - *Dropout:* randomly remove hidden units  
  - *Data Augmentation / Noise Injection:* increase robustness

---

### Residual Networks (ResNets)
Deep networks (>19 layers) suffer from **vanishing gradients** and degraded performance.  
**Residual (skip) connections** solve this by adding inputs back to outputs:

h_l = h_(l−1) + f_phi_l(h_(l−1))

This provides an identity shortcut for signal flow.

- **Effect:** Enables very deep training and stable gradients  
- **Batch Normalization:** Normalizes activations per batch to prevent exploding/vanishing values and stabilize training.

---

### Exercises
Hands-on work included implementing **Feed-Forward Neural Networks (FFNNs)** in PyTorch and exploring **regularization** and **residual blocks** (Notebooks 3.1–3.4).

## Week 4: Convolutional Neural Networks (CNNs)

This week introduced **Convolutional Neural Networks (CNNs)** architectures built for structured data like images.

---

### Motivation: Why Not FFNNs?
Standard **Feed-Forward Networks (FFNNs)** struggle with images due to:

1. **Too Many Parameters:**  
   Example: a 1000×1000 image → about 10⁶ weights in the first layer.  
2. **No Locality:** Treats all pixels independently, ignoring spatial relations.  
3. **No Translation Equivariance:** Must relearn features (like edges) at every position.

---

### CNN Core Ideas
CNNs solve these issues using **local connectivity** and **parameter sharing**.

- **Local Connectivity:** Each neuron connects only to a small region (the **receptive field**).  
- **Parameter Sharing:** The same filter (kernel) slides over the input, detecting patterns everywhere.  
  → Makes CNNs **translation-equivariant** and much more efficient.

---

### Key Components
- **Convolutional Layers:** Apply filters to local regions, add bias, and use nonlinear activations (e.g., ReLU).  
  Multiple filters → multiple **feature maps**.  
- **Pooling Layers:** Downsample spatial size; **Max Pooling** adds partial translation invariance.  
- **Deep Structure:** Stacked layers expand receptive fields, capturing broader context.  
- **Final Layers:** Flattened outputs go through fully connected layers for final prediction (e.g., classification).

CNNs introduce a strong **inductive bias** for image data, improving generalization compared to FFNNs.  
Modern CNNs favor **small filters (3×3)**, **deeper networks**, and sometimes **residual connections** for stable training.

---

### Exercises
Hands-on tasks (Notebooks 4.1–4.3) included implementing **CNNs in PyTorch**, training on **MNIST** and **CIFAR-10**, and exploring **transfer learning**.


## Week 5: Recurrent Neural Networks (RNNs)

This week introduced **Recurrent Neural Networks (RNNs)** models designed for **sequential data** (x = x₁,…,x_T) where sequence length T can vary.

---

### Motivation & Core Structure
**Feed-Forward Networks (FFNNs)** are limited for sequential data because they:
- Require fixed input length (padding)
- Waste parameters on short sequences
- Don’t share weights across time
- Can’t generalize to longer sequences

RNNs solve this by **reusing the same weights** across time steps and maintaining a **hidden state** h(t) that stores past information.

Recurrence relation:  
h(t) = σ(Wₓx(t) + Wₕh(t−1) + bₓ + bₕ)

**Key properties:**
- Handle variable-length sequences  
- Share parameters efficiently  
- Capture long-range dependencies  
- Respect causality (y(t) depends on x(≤t))  
- Generalize to unseen sequence lengths  

Training uses **Maximum Likelihood Estimation (MLE)** by minimizing **negative log-likelihood**, assuming conditional independence across time.

---

### Challenges & LSTMs
RNNs often suffer from **vanishing/exploding gradients** because repeated multiplication by weights (W) either shrinks or amplifies gradients exponentially.  

**LSTMs (Long Short-Term Memory)** fix this by introducing a **cell state C_t** that preserves long-term information through **gates**:
- Forget gate f_t  
- Input gate i_t  
- Candidate cell state C̃_t  
- Output gate o_t  

These control information flow, stabilizing gradient propagation. LSTMs are key precursors to **Transformers**.

---

### Variations & Alternatives
- **Stacked RNNs:** Multiple RNN layers → deeper models  
- **Bidirectional RNNs:** Process sequences forward & backward (uses past + future context, but not causal)  
- **Echo State Networks (ESNs):** Use a fixed random “reservoir”, only output weights are trained, reducing gradient issues

---

### Exercises
Hands-on practice in *Notebook 5.1*:
- Implemented **Vanilla RNNs** with Nanograd  
- Implemented **LSTMs** in both Nanograd and PyTorch

## Week 6: Transformers

This week introduced the **Transformer** an architecture specialized for **sequential data** and foundational for modern large language models (LLMs) and vision tasks.  
The focus was on the **Attention mechanism**, the core of the Transformer layer.

---

### Attention Mechanism
Originally developed to improve **RNN encoder-decoder** models (e.g., for translation), **Attention** allows a model to focus on relevant parts of the input sequence.

- **Relevance scores:**  
  e_j = f_a(z*, h_j) — often a dot product: e_j = z* · h_j  
- **Weights:**  
  a_j = Softmax(e_j)  
- **Context vector:**  
  c* = Σ_j a_j * h_j  

The core operation is **Scaled Dot-Product Attention**:  
Attention(Q, K, V) = Softmax((QKᵀ) / √D_k) * V

#### Multi-Head & Self-Attention
- **Multi-Head Attention:** Runs several attention layers in parallel, capturing information from multiple representation subspaces.  
- **Self-Attention:** Q, K, and V come from the same input, connects all positions in a sequence, enabling **long-range dependencies** and **parallel computation** (unlike RNNs).

---

### Transformer Layer Architecture
A Transformer layer removes the need for recurrence and includes:

1. **Multi-Head Self-Attention**  
2. **Residual Connections** (as in ResNets)  
3. **Layer Normalization** - normalizes per sample, not per batch  
4. **Feed-Forward Network (FFN)** - fully connected layer  

**Positional Encoding** is added to retain order information since self-attention alone is permutation-invariant.

---

### Transformer Model Types
- **Encoder-only:** e.g., BERT - trained with self-supervision (masked tokens)  
- **Decoder-only:** e.g., GPT - uses causal self-attention for next-token prediction  
- **Encoder-Decoder:** e.g., translation models combining both parts

---

### Exercises
Practical work (*Notebo*


## Project Idees
Multi-modal registration to mri. 
Dataset:
https://www.kaggle.com/datasets/grantmcnatt/mri-and-pet-dice-similarity-dataset?utm_source=chatgpt.com
https://www.kaggle.com/datasets/29c3607295965ebb030f2d158fec487412d84c82528dd44f8ef956aef35541aa
https://www.med.upenn.edu/cbica/brats2020/data.html?utm_source=chatgpt.com
https://sites.wustl.edu/oasisbrains/?utm_source=chatgpt.com
https://www.kaggle.com/datasets/purnimakumarrr/adhd200-preprocessed-anatomical-dataset
https://www.synapse.org/Synapse:syn64153130/wiki/
