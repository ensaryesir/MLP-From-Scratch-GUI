# MLP From Scratch with GUI

This project implements a Multi-Layer Perceptron (MLP) and other neural network architectures from scratch using Python, without relying on deep learning frameworks like TensorFlow or PyTorch. It features a comprehensive Graphical User Interface (GUI) built with `customtkinter` for visualizing training, decision boundaries, and latent spaces.

## Technical Architecture

This project implements three fundamental neural network algorithms with their mathematical foundations manually coded.

### 1. Perceptron
*   **Update Rule:**
    ```
    w = w + η * (y_true - y_pred) * x
    ```
*   **Activation:** Step function 
*   **Capabilities:** Binary and Multi-class classification (Winner-Takes-All).

### 2. Delta Rule (Adaline)
*   **Loss Function (MSE):**
    ```
    MSE = (1/n) * Σ(y_true - y_pred)²
    ```
*   **Gradient:**
    ```
    ∂L/∂w = -(2/n) * X^T * (y_true - y_pred)
    ```
*   **Activation:** Linear
*   **Optimization:** Gradient Descent.

### 3. Multi-Layer Perceptron (MLP)
Implemented with fully manual forward and backward propagation steps.

#### Forward Propagation
```
Z^[l] = A^[l-1] @ W^[l] + b^[l]
A^[l] = activation(Z^[l])
```

#### Backpropagation
```
dZ^[L] = A^[L] - Y  (Output Layer)
dW^[l] = (1/m) * (A^[l-1])^T @ dZ^[l]
db^[l] = (1/m) * Σ dZ^[l]
dZ^[l-1] = (dZ^[l] @ (W^[l])^T) * σ'(Z^[l-1])
```

#### Activation Functions
```
ReLU:    f(x) = max(0, x)
Tanh:    f(x) = tanh(x)
Sigmoid: f(x) = 1 / (1 + e^-x)
Softmax: f(x_i) = e^x_i / Σ e^x_j
```

#### Loss & Optimization
*   **Cross-Entropy:**
    ```
    L = -(1/m) * Σ Σ y_true * log(y_pred)
    ```
*   **L2 Regularization:**
    ```
    L_reg = (λ/2m) * Σ||W||²
    ```
*   **Optimization:** Mini-batch Gradient Descent.

## Features

The application supports two main modes of operation: **Manual Data Interaction** and **MNIST Dataset Analysis**.

### 1. Manual Mode (2D Playground)
In this mode, users can interactively create datasets on a 2D plane and train models to verify their learning capabilities.

#### Classification
*   **Usage:** Add multiple classes, click on the canvas to add data points for each class, and train the model.
*   **Visualization:** The application visualizes the decision boundaries in real-time, showing how the network separates different classes.
*   **Models:** Supports Perceptron (Single Layer), Delta Rule, and MLP.

![Manual Classification Screenshot](PLACEHOLDER_FOR_MANUAL_CLASSIFICATION_IMAGE_HERE)
*(2D classification decision boundaries)*

#### Regression
*   **Usage:** Select "Regression" task (implied by 1 class output or specific configuration), add points to define a function $y = f(x)$.
*   **Visualization:** Visualizes the regression curve fitting the data points.
*   **Algorithms:** Uses Delta Rule (Adaline) or MLP for function approximation.

![Manual Regression Screenshot](PLACEHOLDER_FOR_MANUAL_REGRESSION_IMAGE_HERE)
*(Regression curve)*

---

### 2. MNIST Mode (Digit Recognition)
This mode applies the neural networks to the classic MNIST handwritten digit dataset.

#### MLP Classification
*   Trains a Multi-Layer Perceptron to classify handwritten digits (0-9).
*   Displays training error and confusion matrix/performance metrics.

#### Autoencoder & Feature Extraction
*   **Autoencoder:** Trains an unsupervised Autoencoder to compress images into a low-dimensional latent space and reconstruct them.
*   **Hybrid Model:** Uses the pre-trained Encoder as a feature extractor, feeding the compressed representation into a smaller MLP for classification.
*   **Visualizations:**
    *   **Reconstruction:** Shows original vs. reconstructed digit images.
    *   **Latent Space:** Visualizes the 2D projection of the bottleneck layer, showing how digits cluster in the compressed space.

![MNIST Analysis Screenshot](PLACEHOLDER_FOR_MNIST_IMAGE_HERE)
*(MNIST reconstruction or latent space)*

## Installation & Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ensaryesir/MLP-From-Scratch-GUI.git
    cd MLP-From-Scratch-GUI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python main.py
    ```

## Project Structure
*   `algorithms/`: Core implementations of Perceptron, Delta Rule, MLP, and Autoencoder.
*   `gui/`: User interface components and visualization logic.
*   `utils/`: Matrix operations, activation functions, and data handlers.
*   `config/`: Hyperparameters and configuration defaults.