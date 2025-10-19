# Deep_learning
By Holger Max Fløe Lyng, s214776
This is a repository for 02456 deep learing course at DTU. I am a student in my master on my 9th semester.

## Lecture 1 - Neural Networks
In this lecture we learn basic neural network (\textbf{nn}) this i a network containing only one hidden layer (K) and arbitrarely many hidden units (D). We get introduced to deep nn, nn containing more than one hidden layer (K), stocastic gradient descent, backpropagation and loss function. We learned how to calculate loss function for classification and normal gaussian. we learned how gradient descent and backpropagation works and how to generalize it. See (call 1.1 FNN Pen and paper) for exercises.

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
- Random weights: \( \phi_0 \sim \mathcal{N}(0, \sigma_0^2) \)
- **He Initialization** for ReLU:
  \[
  \sigma_0 = \sqrt{\frac{2}{D}}
  \]
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
