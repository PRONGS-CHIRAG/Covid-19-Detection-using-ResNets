# 🦠 Covid-19 Detection using ResNets

## 📚 Overview
This project leverages a pre-trained **ResNet-18** model to detect **Covid-19**, **Viral Pneumonia**, and **Normal** cases from chest X-ray images. The model is fine-tuned using PyTorch on a custom dataset, and it provides real-time performance monitoring and visualization.

## 🎯 Objectives
- Classify chest X-ray images into **Normal**, **Covid**, and **Viral Pneumonia** categories.
- Fine-tune **ResNet-18**, a robust deep learning model.
- Provide real-time training feedback and prediction visualization.
- Build an interpretable and modular pipeline using PyTorch.

---

## 🏗️ Project Structure
```
Covid19_Detection/
│
├── Covid19_Detection.ipynb       # Main Jupyter notebook for training, testing, and visualization
├── utils.py                      # Utility functions and custom Dataset class

```

---

## ⚙️ Features

### ✅ Custom Dataset Class (`chestxraydataset`)
- Loads `.png` files from separate folders.
- Applies image transformations using `torchvision.transforms`.
- Dynamically samples and returns `(image_tensor, class_label)` pairs.

### ✅ Training Pipeline
- Uses pre-trained **ResNet-18** from `torchvision.models`.
- Optimized with **CrossEntropyLoss** and **Adam** optimizer.
- Training loop with periodic validation and accuracy reporting.
- **Early stopping** if validation accuracy exceeds 95%.

### ✅ Visualization
- Real-time display of **true vs predicted labels**.
- Color-coded results (✔ Green: correct, ✘ Red: incorrect).
- Uses `matplotlib` for image visualization.

---

## 🧠 Model Architecture
- **Backbone**: ResNet-18 (pre-trained on ImageNet).
- **Modifications**: Final fully connected layer adapted for 3 output classes.
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam Optimizer.

---

## 📈 Evaluation
- Accuracy and Loss tracked every 20 steps.
- Visual confirmation using a batch of predictions.
- Helps identify overfitting or class confusion early.

---

## 🔍 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Covid19_Detection.git
cd Covid19_Detection
```

### 2. Install Requirements
```bash
pip install torch numpy matplotlib PIL shutil torchvision
```

### 3. Run the Notebook
Open the Jupyter notebook and execute `Covid19_Detection.ipynb` step by step:
```bash
jupyter notebook Covid19_Detection.ipynb
```

---

## 🧪 Key Functions

| Function         | Description |
|------------------|-------------|
| `chestxraydataset` | Custom PyTorch dataset class |
| `train(epochs)`   | Trains the model for given epochs |
| `show_preds()`    | Visualizes model predictions |
| `show_images()`   | Plots images with labels and predictions |

---


## 📎 Dependencies
- `torch`, `torchvision`
- `numpy`, `matplotlib`
- `PIL` (Pillow)

---


## 📄 License
This project is licensed under the MIT License.

---

## ✍️ Author
 Chirag N Vijay

