<div align="center">

# MLP From Scratch with GUI

A powerful neural network visualization and training tool built entirely from scratch using pure NumPy. Train, visualize, and experiment with Multi-Layer Perceptrons, Autoencoders, and classic learning algorithms through an intuitive GUI.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## ✨ Key Features

### 🧠 **Neural Network Algorithms**
- **Multi-Layer Perceptron (MLP)** - Fully customizable architecture with backpropagation
- **Single Layer Perceptron** - Classic binary/multi-class classifier
- **Delta Rule (Adaline)** - Gradient descent-based learning
- **Autoencoder** - Unsupervised feature extraction and dimensionality reduction
- **Hybrid Autoencoder-MLP** - Two-stage training with encoder reuse

### 🎨 **Interactive Visualization**
- **Real-time decision boundaries** - Watch your network learn
- **Loss/Error graphs** - Track convergence across epochs
- **Reconstruction visualization** - See autoencoder outputs (10 digits in 4×5 grid)
- **Training/Test split** - Separate visualization for validation
- **Modern dark theme** - Built with CustomTkinter

### ✍️ **MNIST Handwriting Tester**
- **Draw your own digits** - 280×280 canvas for easy drawing
- **Real-time prediction** - Instant classification with confidence
- **MNIST preprocessing** - Automatic centering, resizing, and normalization

### 💾 **Model Persistence**
- **Save/Load Models** - Export trained models as `.pkl` files
- **Encoder Save/Load** - Reuse trained autoencoders across sessions
- **Two-stage workflow** - Train encoder → Save → Train MLP separately

### 📊 **Datasets**
- **Manual 2D Playground** - Click to create custom datasets
- **MNIST** - Handwritten digit recognition (60k train, 10k test)
- **Built-in Presets**:
  - Classification: XOR, Circles, Moons, Blobs
  - Regression: Sine, Parabola, Linear, Absolute

### ⚙️ **Advanced Configuration**
- **Flexible architecture** - Define any layer structure (e.g., 784-256-128-10)
- **Activation functions** - ReLU, Sigmoid, Tanh, Softmax
- **Hyperparameter tuning** - Learning rate, batch size, epochs, momentum
- **Stopping criteria** - Converge on min error or max epochs
- **L2 Regularization** - Prevent overfitting
- **Momentum** - Accelerate training with momentum factor

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ensaryesir/MLP-From-Scratch-GUI.git
cd MLP-From-Scratch-GUI

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

---

## 📖 Usage Guide

### 1️⃣ **Manual Mode** (2D Playground)

Perfect for understanding how neural networks learn decision boundaries.

#### Classification
1. Select **"Classification"** task
2. Select **"Manual"** dataset
3. Choose a model (Perceptron, Delta Rule, or MLP)
4. Add classes and click on canvas to create data points
5. Configure hyperparameters
6. Press **"START TRAINING"**

**Built-in Presets:**
- **XOR Problem** - Classic non-linearly separable dataset
- **Circles** - Concentric circles classification
- **Moons** - Two interleaving half-circles
- **Blobs** - Gaussian clusters

#### Regression
1. Select **"Regression"** task
2. Select **"Manual"** dataset
3. Choose Delta Rule or MLP
4. Add points to define target function
5. Train and watch the curve fit

**Built-in Presets:**
- Sine Wave
- Parabola
- Linear
- Absolute Value

---

### 2️⃣ **MNIST Mode** (Digit Recognition)

Train on 60,000 handwritten digits with professional visualizations.

#### Standard MLP Training
1. Select **"MNIST"** dataset
2. Choose **"Multi-Layer MLP"**
3. Configure architecture (e.g., `784,256,10`)
4. Set stopping criteria (default: Min Error 0.01)
5. Press **"START TRAINING"**
6. Monitor error graphs in real-time

**After Training:**
- **Save Model** - Export trained weights
- **Load Model** - Import previously saved model
- **Test Handwriting** - Draw digits and get predictions

#### Autoencoder Workflow (Two-Stage)

**Stage 1: Train Encoder**
1. Select **"MNIST"** dataset
2. Choose **"Autoencoder-Based MLP"**
3. Configure encoder architecture (e.g., `784,128`)
4. Press **"START TRAINING"**
5. Wait for Stage 1 completion popup
6. **Save Encoder** for reuse
7. Press **"START TRAINING"** again for Stage 2

**Stage 2: Train MLP**
- Uses pre-trained encoder as feature extractor
- Trains smaller MLP on compressed features (e.g., `128,64,10`)
- Faster training with better generalization

**Visualization:**
- **Autoencoder Error** - Reconstruction loss graph
- **MLP Error** - Classification loss graph
- **Reconstruction** - View 10 original vs reconstructed digits

---

## 🏗️ Technical Architecture

### Core Algorithms

#### 1. **Perceptron**
```
Update Rule: w = w + η * (y_true - y_pred) * x
Activation: Step function
Use Case: Binary/Multi-class classification
```

#### 2. **Delta Rule (Adaline)**
```
Loss (MSE): L = (1/n) * Σ(y_true - y_pred)²
Gradient: ∂L/∂w = -(2/n) * X^T * (y_true - y_pred)
Optimization: Gradient Descent
Use Case: Regression, Linear classification
```

#### 3. **Multi-Layer Perceptron (MLP)**

**Forward Propagation:**
```
Z[l] = A[l-1] @ W[l] + b[l]
A[l] = activation(Z[l])
```

**Backpropagation:**
```
dZ[L] = A[L] - Y  (Output layer)
dW[l] = (1/m) * A[l-1]^T @ dZ[l]
db[l] = (1/m) * Σ dZ[l]
dZ[l-1] = (dZ[l] @ W[l]^T) * σ'(Z[l-1])
```

**Activation Functions:**
```python
ReLU:    f(x) = max(0, x)
Tanh:    f(x) = tanh(x)
Sigmoid: f(x) = 1 / (1 + e^-x)
Softmax: f(x_i) = e^x_i / Σ e^x_j
```

**Loss & Optimization:**
```
Cross-Entropy: L = -(1/m) * Σ Σ y_true * log(y_pred)
L2 Regularization: L_reg = (λ/2m) * Σ||W||²
Momentum: v = β*v + (1-β)*∇W
```

#### 4. **Autoencoder**
- **Encoder**: Compresses input to latent representation
- **Decoder**: Reconstructs input from latent space
- **Training**: MSE between input and reconstruction
- **Feature Extraction**: Use encoder weights for classification

---

## 📁 Project Structure

```
MLP-From-Scratch-GUI/
│
├── algorithms/          # Neural network implementations
│   ├── mlp.py          # Multi-Layer Perceptron
│   ├── perceptron.py   # Single-layer Perceptron
│   ├── delta_rule.py   # Adaline algorithm
│   ├── autoencoder.py  # Autoencoder implementation
│   └── mlp_with_encoder.py  # Hybrid model
│
├── gui/                # User interface
│   ├── control_panel.py       # Hyperparameter controls
│   ├── visualization_frames.py # Plots and graphs
│   ├── training_manager.py    # Training orchestration
│   └── handwriting_tester.py  # MNIST drawing canvas
│
├── utils/              # Helper functions
│   ├── data_handler.py # Dataset management
│   ├── load_mnist.py   # MNIST loader
│   └── activations.py  # Activation functions
│
├── config/             # Configuration
│   └── default_hyperparams.py  # Default settings
│
├── dataset/            # Data storage
│   └── MNIST/         # MNIST binary files
│
├── weights/           # Saved models (auto-created)
│
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 🎯 Use Cases

- **Educational** - Learn neural network fundamentals from scratch
- **Experimentation** - Test architectures and hyperparameters
- **Visualization** - Understand how networks learn decision boundaries
- **Research** - Prototype custom learning algorithms
- **Teaching** - Demonstrate ML concepts interactively

---


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


## 📧 Contact

**Ensar Yesir** - [@ensaryesir](https://github.com/ensaryesir)

Project Link: [https://github.com/ensaryesir/MLP-From-Scratch-GUI](https://github.com/ensaryesir/MLP-From-Scratch-GUI)

---

⭐ **Star this repo if you find it useful!**