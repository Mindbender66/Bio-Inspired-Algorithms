# 🧠 Autism Detection Using Swarm Intelligence & Deep Learning

> A comprehensive comparative study combining **nature-inspired Swarm Intelligence algorithms** with classical **Machine Learning** and **Deep Learning** models for Autism Spectrum Disorder (ASD) detection.  
> Two parallel tracks: **Tabular behavioral data** (PSO & Firefly) and **Facial image classification** (ACO & ABC) across CNN, VGG16, and ResNet50 architectures.

---

## 🩺 Problem Statement

Autism Spectrum Disorder (ASD) affects millions of children worldwide, and **early detection is critical** for effective intervention. Manual diagnosis relies on behavioral assessments and expert evaluation, which can be delayed or inaccessible.

This project automates ASD detection using two complementary data modalities:

- **Behavioral questionnaire responses** (Toddler ASD tabular dataset)
- **Facial image analysis** (Autistic vs Non-Autistic image classification)

Rather than standard model training, this project investigates whether **Swarm Intelligence algorithms** can automatically find better feature subsets and hyperparameter configurations — replacing tedious manual tuning with nature-inspired search.

---

## 🏷️ Classes

| Track | Classes |
|-------|---------|
| Tabular (Track A) | `Yes` — ASD traits present / `No` — ASD traits absent |
| Image (Track B) | `Autistic` / `Non-Autistic` |

---

## 📦 Datasets

### Dataset 1 — Toddler Autism Dataset (Tabular)

- **Format:** CSV (structured tabular data)
- **Task:** Binary classification of ASD traits
- **Target Column:** `Class/ASD Traits`
- **Features:** A1–A10 behavioral scores, Age in months, Sex, Ethnicity, Jaundice history, Family ASD history, Qchat-10-Score

#### 📂 Feature Overview

| Feature | Type | Description |
|---------|------|-------------|
| A1 – A10 | Binary | Q-CHAT behavioral screening questions |
| Age_Mons | Numeric | Age of toddler in months |
| Qchat-10-Score | Numeric | Total autism screening score |
| Sex | Categorical | Male / Female |
| Ethnicity | Categorical | Ethnic background |
| Jaundice | Binary | Jaundice at birth (Yes/No) |
| Family_mem_with_ASD | Binary | Family member with ASD (Yes/No) |
| Class/ASD Traits | Binary | **TARGET** — Yes / No |

---

### Dataset 2 — Facial Images for Autism Detection (Image)

- **Format:** Image folders (`train/`, `test/`, `valid/`)
- **Classes:** `Autistic` / `Non-Autistic`
- **Image Size:** Resized to `32×32` (swarm versions) or `150×150` / `224×224` (baseline)
- **Loading:** OpenCV `cv2.imread()` + `cv2.resize()`
- **Normalization:** Pixel values ÷ 255.0 → scaled to [0, 1]

#### 📂 Dataset Structure

```bash
dataset/
├── train/
│   ├── Autistic/
│   └── Non_Autistic/
├── test/
│   ├── Autistic/
│   └── Non_Autistic/
└── valid/
    ├── Autistic/
    └── Non_Autistic/
```

---

## 🐝 Swarm Intelligence Algorithms

### 🔵 Particle Swarm Optimization (PSO)
*Inspired by bird flocking and fish schooling*

Each particle = a candidate feature selection mask. Particles converge toward the best-performing feature subset by balancing personal memory and global swarm knowledge.

- **Used in:** `particle_swarm_ann.py`
- **Role:** Feature selection on PCA-reduced tabular features
- **Library:** `pyswarm`
- **Key params:** `swarmsize=20`, `maxiter=50`

---

### 🟢 Ant Colony Optimization (ACO)
*Inspired by ant pheromone trail communication*

Ants evolve a population of binary configurations using DEAP genetic operators (crossover, mutation, tournament selection) — capturing pheromone-like convergence without explicit pheromone matrices.

- **Used in:** `ant.py`
- **Role:** Hyperparameter optimization for CNN & VGG on facial images
- **Library:** `DEAP`
- **Key operators:** `cxTwoPoint`, `mutFlipBit(indpb=0.05)`, `selTournament(tournsize=3)`

---

### 🟩 Artificial Bee Colony (ABC)
*Inspired by honey bee foraging behaviour*

Three bee types work in concert: **Employed Bees** exploit known solutions, **Onlooker Bees** focus on the best sources, and **Scout Bees** escape local optima through random exploration.

- **Used in:** `artificial_bee_colony.py`
- **Role:** Hyperparameter optimization for CNN, VGG, and ResNet50
- **Key params:** `max_iterations=20`, `population_size` variable

| Bee Type | Behaviour | Role |
|----------|-----------|------|
| EmployedBee | Flip one random bit | Exploit current solutions |
| OnlookerBee | Follow best employed bee | Focused exploitation |
| ScoutBee | Random new solution | Global exploration |

---

### 🟠 Firefly Algorithm (FA)
*Inspired by bioluminescent firefly attraction*

Dimmer fireflies move toward brighter ones (better solutions). Attractiveness decreases with distance, controlled by parameters `alpha` (randomization) and `gamma` (light absorption).

- **Used in:** `fire_fly.py`
- **Role:** Feature selection for ANN, RF, KNN, and GaussianNB on tabular data
- **Key params:** `alpha=0.5`, `beta_min=0.2`, `gamma=1.0`, `max_iter=25–50`

---

## 🏗️ System Architecture

### Track A — Tabular Data Pipeline

```text
Load CSV
    │
    ▼
EDA (histplot, boxplot, countplot, pointplot)
    │
    ▼
Label Encoding + StandardScaler
    │
    ▼
PCA (n_components = 10 or 15)
    │
    ▼
Swarm Optimization (PSO / Firefly)
→ Binary feature mask output
    │
    ▼
Train/Test Split (80/20)
    │
    ▼
GridSearchCV (5-fold CV)
    │
    ▼
Train Best Classifier (ANN / RF / KNN / GaussianNB)
    │
    ▼
Evaluate → accuracy, classification report, confusion matrix
```

---

### Track B — Image Data Pipeline

```text
Load Images (OpenCV → resize → normalize)
    │
    ▼
Swarm Optimization (ACO / ABC)
→ Searches {optimizer, dropout_rate}
    │
    ▼
Fitness Evaluation (3-epoch proxy training)
    │
    ▼
Extract Best Hyperparameters
    │
    ▼
Train Best Model (10 epochs)
CNN / VGG16 / ResNet50
    │
    ▼
Evaluate + Save (.h5)
```

---

## 🧠 Deep Learning Models

### CNN (Custom)
```text
Input (32×32×3 or 150×150×3)
    → Conv2D(32) + MaxPool
    → Conv2D(64) + MaxPool
    → Flatten
    → Dense(128, ReLU)
    → Dropout
    → Dense(num_classes, Softmax)
```

### VGG16
```text
Block 1: Conv2D(64) × 2 + MaxPool
Block 2: Conv2D(128) × 2 + MaxPool
Block 3: Conv2D(256) × 3 + MaxPool
    → Flatten
    → Dense(512) + Dropout × 2
    → Dense(num_classes, Softmax)
```
*With transfer learning (baseline): frozen ImageNet weights + custom head*

### ResNet50
```text
ResNet50 Base (frozen, ImageNet weights)
    → GlobalAveragePooling2D
    → Dense(512, ReLU)
    → Dropout
    → Dense(num_classes, Softmax)
```

---

## ⚙️ Hyperparameter Optimization Strategy

| Layer | Method | What Is Optimized |
|-------|--------|-------------------|
| Outer (Track A) | PSO / Firefly | Which PCA features to include |
| Inner (Track A) | GridSearchCV (CV=3–5) | Classifier hyperparameters |
| Outer (Track B) | ACO / ABC | Optimizer choice + dropout rate |
| Inner (Track B) | Manual grid | Batch size, epochs |

### GridSearchCV Parameter Spaces

| Classifier | Key Parameters |
|------------|----------------|
| MLP/ANN | hidden_layer_sizes, activation, solver, alpha, learning_rate |
| Random Forest | n_estimators, max_features, max_depth, criterion |
| KNN | n_neighbors, weights, metric |
| GaussianNB | No tuning needed (closed-form solution) |

---

## 🤖 Machine Learning Classifiers (Track A)

| Classifier | Key Strength | Notes |
|------------|-------------|-------|
| MLP / ANN | Universal approximator | sklearn MLPClassifier, not Keras |
| Random Forest | Robust to overfitting | Ensemble of decision trees |
| K-Nearest Neighbors | Non-parametric | No model learned — memorizes data |
| Gaussian Naive Bayes | Fast, no tuning | Assumes feature independence |

---

## 📊 Evaluation Metrics

| Metric | Formula | Priority |
|--------|---------|----------|
| Accuracy | Correct / Total | General performance |
| Precision | TP / (TP + FP) | False positive cost |
| **Recall** | **TP / (TP + FN)** | **⭐ Most critical for ASD screening** |
| F1 Score | 2 × (P × R) / (P + R) | Imbalanced classes |
| Confusion Matrix | TP/TN/FP/FN table | Detailed error analysis |

> ⚠️ **Recall is the primary metric.** A False Negative (missed ASD case) means a child misses critical early intervention — far more harmful than a False Positive.

---

## 📈 Visualizations Included

| Visualization | Purpose |
|---------------|---------|
| Confusion Matrix Heatmap | Exact TP/FP/TN/FN per class |
| Training & Validation Accuracy Curves | Overfitting detection |
| Training & Validation Loss Curves | Convergence monitoring |
| PCA Explained Variance Bar Chart | Component selection |
| Correlation Heatmap | Feature relationship analysis |
| Selected Features Bar Chart | PSO/Firefly selection output |
| Class Distribution Count Plot | Class balance check |
| Scatter / Box / Violin / Swarm Plots | EDA (fire_fly.py) |
| Prediction Samples Grid | Visual model verification |

---

## 🗂️ Project Structure

```bash
autism-detection-swarm/
│
├── particle_swarm_ann.py          # Track A: PSO + ANN on tabular data
├── fire_fly.py                    # Track A: Firefly + ANN/RF/KNN/NB on tabular data
├── without_swarm_and_fire_fly.py  # Track A: Baseline (GridSearch only)
│
├── ant.py                         # Track B: ACO + CNN/VGG on facial images
├── artificial_bee_colony.py       # Track B: ABC + CNN/VGG/ResNet on facial images
├── without_ant_and_bee.py         # Track B: Baseline (pretrained models, no swarm)
│
├── README.md                      # Project documentation
│
├── saved_models/
│   ├── CNN_Ant.h5                 # CNN trained with ACO
│   ├── CNN_Artificial.h5          # CNN trained with ABC
│   ├── VGG_Artificial.h5          # VGG trained with ABC
│   ├── CNN.h5                     # CNN baseline
│   ├── VGG.h5                     # VGG baseline
│   ├── RESNET.h5                  # ResNet50 baseline
│   ├── ANNmodelFF.pkl             # ANN trained with Firefly
│   ├── RFmodelFF.pkl              # Random Forest with Firefly
│   ├── KNNmodelFF.pkl             # KNN with Firefly
│   └── NBmodelFF.pkl              # Naive Bayes with Firefly
│
└── dataset/
    ├── toddler_autism.csv         # Tabular dataset
    └── facial_images/
        ├── train/
        ├── test/
        └── valid/
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone <your-repository-link>
cd autism-detection-swarm
```

---

### 2️⃣ Install Dependencies

```bash
pip install tensorflow scikit-learn numpy pandas opencv-python matplotlib seaborn pyswarm deap
```

---

### 3️⃣ Prepare Datasets

**Tabular dataset:** Place the CSV file and update the path:
```python
data = pd.read_csv('/content/drive/MyDrive/toddler_autism.csv')
```

**Image dataset:** Organize folders as shown in the structure above and update:
```python
train_path = '/content/drive/MyDrive/dataset/train'
test_path  = '/content/drive/MyDrive/dataset/test'
valid_path = '/content/drive/MyDrive/dataset/valid'
```

---

### 4️⃣ Run the Scripts

**Track A — Tabular with PSO:**
```bash
python particle_swarm_ann.py
```

**Track A — Tabular with Firefly:**
```bash
python fire_fly.py
```

**Track B — Images with ACO:**
```bash
python ant.py
```

**Track B — Images with ABC:**
```bash
python artificial_bee_colony.py
```

**Baseline comparisons (no swarm):**
```bash
python without_swarm_and_fire_fly.py
python without_ant_and_bee.py
```

---

## 🔍 Single Image Prediction (Track B)

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('CNN_Artificial.h5')
image = cv2.imread('sample.jpg')
image = cv2.resize(image, (32, 32))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
predicted_class = np.argmax(prediction)
print("Autistic" if predicted_class == 0 else "Non-Autistic")
```

---

## 💡 Key Design Decisions

- **Two-track architecture** separates tabular and image modalities cleanly
- **PSO uses negative accuracy** as its objective — PSO minimises, so `-accuracy` = maximise accuracy
- **32×32 images in swarm versions** — enables thousands of candidate evaluations without GPU timeout
- **PCA before swarm selection** compresses the feature search space from 2ⁿ to a tractable size
- **DEAP framework** for ACO — captures evolutionary search without explicit pheromone matrices
- **Three bee classes** in ABC map directly to the biological exploration/exploitation framework
- **Transfer learning** (VGG16, ResNet50) for baseline — leverages ImageNet features without large ASD datasets

---

## 🔬 Future Improvements

- [ ] Use full-resolution images (150×150) inside swarm evaluation with smaller populations
- [ ] Add early stopping during swarm proxy training to reduce noise
- [ ] Address class imbalance with `class_weight` or SMOTE oversampling
- [ ] Add 5-fold cross-validation for final model robustness
- [ ] Expand ABC/ACO search space to include learning rate and number of layers
- [ ] Implement GradCAM for facial region interpretability
- [ ] Deploy via Streamlit / Gradio for clinical demo
- [ ] Export models to ONNX / TFLite for edge deployment
- [ ] Compare against standard AutoML benchmarks (e.g., TPOT, Auto-sklearn)

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| TensorFlow / Keras | CNN, VGG16, ResNet50 training and evaluation |
| scikit-learn | MLPClassifier, RF, KNN, GaussianNB, GridSearchCV, PCA, StandardScaler, metrics |
| DEAP | ACO via distributed evolutionary algorithms |
| pyswarm | Particle Swarm Optimization |
| NumPy | Array operations, masking, one-hot handling |
| pandas | CSV loading and exploratory data analysis |
| OpenCV (`cv2`) | Image loading and resizing |
| matplotlib | Training curves, bar charts, prediction grids |
| seaborn | Heatmaps, countplots, violin/swarm/strip plots |
| Google Colab + Drive | GPU-accelerated training and dataset storage |
| joblib / pickle | Model serialization for sklearn models |

---

## ⚠️ Disclaimer

> This project is intended for **educational and research purposes only**.  
> It is **not** a clinical diagnostic tool and should **not** replace professional medical evaluation or a formal ASD assessment by a qualified specialist.

---

## 🙋 Author

**Valmiki Sarath Kumar**

---

## 📄 License

This project is open source and available under the **MIT License**.
