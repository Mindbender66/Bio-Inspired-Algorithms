# 🧠 Brain Tumor Classification — MRI Image CNN

> A deep learning project using a custom **4-layer Convolutional Neural Network** to classify brain MRI scans into three tumor types: **Glioma**, **Meningioma**, and **Pituitary**. Trained using the **RMSprop optimizer** over 50 epochs with data augmentation on real clinical MRI images.

---

## 🩺 Problem Statement

Brain tumors are among the most dangerous and life-threatening conditions, requiring accurate diagnosis from MRI scans for effective treatment planning. This project automates the classification of three primary brain tumor types using a CNN — helping reduce diagnostic time and support radiologists.

---

## 🏷️ Classes

| Class | Description |
|-------|-------------|
| `glioma` | Forms a mass (tumor) that presses on brain/spinal cord tissue and causes symptoms |
| `meningioma` | The most common primary brain tumor — ~30% of all brain tumors |
| `pituitary` | Unusual growths in the pituitary gland (roughly pea-sized organ) |

---

## 📦 Dataset

- **Source:** [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Format:** `.jpg` MRI images organized in class folders
- **Input:** Grayscale images resized to **150 × 150**
- **Split:** 70% Train / 30% Test (stratified)

```
BT/
├── glioma/
├── meningioma/
└── pituitary/
```

---

## 🧠 Model Architecture

A custom sequential CNN with **4 convolutional blocks** followed by fully connected dense layers.

```
Input (150 × 150 × 1 — Grayscale)
    │
    ├── Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
    ├── Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
    ├── Conv2D(128, 3×3, relu) → MaxPooling2D(2×2)
    ├── Conv2D(256, 3×3, relu) → MaxPooling2D(2×2)
    │
    ├── Flatten
    ├── Dense(512, relu) → Dropout(0.5)
    ├── Dense(256, relu) → Dropout(0.5)
    │
    └── Dense(3, softmax)  ← Output: glioma / meningioma / pituitary
```

### ⚙️ Training Config

| Parameter | Value |
|-----------|-------|
| Input Shape | 150 × 150 × 1 (grayscale) |
| Optimizer | **RMSprop** (lr = 0.001) |
| Loss | Categorical Crossentropy |
| Epochs | 50 |
| Batch Size | 32 |
| Random Seed | 42 |

> 🔑 **Key difference from the Breast Cancer model:** This model uses **RMSprop** instead of Adam — better suited for noisy MRI gradients due to its adaptive learning rate per parameter.

---

## 🔄 Data Augmentation

Applied only on the training set:

| Technique | Value |
|-----------|-------|
| Rotation | ±20° |
| Width Shift | ±20% |
| Height Shift | ±20% |
| Shear | ±20% |
| Zoom | ±20% |
| Horizontal Flip | ✅ Yes |
| Fill Mode | Nearest |

---

## 📊 Visualizations Included

| Plot | Description |
|------|-------------|
| 📈 Learning Curve | Training vs. Validation Accuracy over 50 epochs |
| 📉 Loss Curve | Training vs. Validation Loss over 50 epochs |
| 🟦 Confusion Matrix | Heatmap of predicted vs. true classes |
| 🌡️ Image Heatmap | Pixel intensity heatmap of a sample scan (index 90) |
| 🔵 Scatter Plot | Pixel intensity comparison between two meningioma scans |
| 📊 Histogram | Grayscale pixel distribution of a meningioma image |

---

## 🗂️ Repository Structure

```
Brain-Tumor-Classification/
│
├── BT_code.ipynb         # Main training notebook
├── BTmodel.h5            # Saved trained model weights
└── README.md             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Mindbender66/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
```

### 2. Download the Dataset
Get the dataset from Kaggle and organize as:
```
BT/
├── glioma/
├── meningioma/
└── pituitary/
```
Update `dataset_path` in the notebook to your local or Google Drive path.

### 3. Install Dependencies
```bash
pip install tensorflow numpy pandas pillow matplotlib seaborn scikit-learn opencv-python
```

### 4. Run in Google Colab (Recommended)
Upload `BT_code.ipynb` to Colab, mount Google Drive, and run all cells with a **GPU runtime**.

---

## 🔍 Single Image Prediction

The notebook includes an end-to-end prediction pipeline for any MRI scan:

```python
image_path = "your_mri_scan.jpg"
test_image = load_preprocess_single_image(image_path)
predictions = model.predict(test_image)
predicted_label = classes[np.argmax(predictions)]
# → 'glioma', 'meningioma', or 'pituitary'
```

**Output messages:**
- 🔴 `glioma` → *"As a glioma grows it forms a mass that can press on brain or spinal cord tissue and cause symptoms."*
- 🟠 `meningioma` → *"Meningioma is the most common primary brain tumor — ~30% of all brain tumors."*
- 🟡 `pituitary` → *"Pituitary tumors are unusual growths in the pituitary gland, an organ about the size of a pea."*

---

## 🧰 Tech Stack

| Library | Purpose |
|---------|---------|
| `TensorFlow / Keras` | Model building & training |
| `Pillow (PIL)` | Image loading & preprocessing |
| `OpenCV (cv2)` | Grayscale histogram analysis |
| `scikit-learn` | Train/test split, metrics, label encoding |
| `matplotlib / seaborn` | Plots & confusion matrix |
| `NumPy / Pandas` | Array operations & data handling |
| `Google Colab` | Training environment (GPU) |

---

## 💡 Key Design Choices

- **Grayscale input** — MRI scans carry diagnostic information in intensity, not color; single-channel reduces noise
- **4 Conv blocks with doubling filters** (32 → 64 → 128 → 256) — hierarchical feature extraction from edges to complex tumor shapes
- **RMSprop optimizer** — chosen for its ability to handle noisy, non-stationary MRI gradient signals
- **Double Dropout(0.5)** — strong regularization to prevent overfitting on medical imaging data
- **Stratified split** — ensures each tumor class is proportionally represented in train/test

---

## 🔬 Comparison With Breast Cancer CNN

| Feature | Brain Tumor (BT) | Breast Cancer (BC) |
|---------|-----------------|-------------------|
| Classes | glioma, meningioma, pituitary | benign, malignant, normal |
| Image Type | MRI (brain) | Ultrasound (breast) |
| Optimizer | **RMSprop** | Adam |
| Architecture | Identical 4-block CNN | Identical 4-block CNN |
| Saved Model | `BTmodel.h5` | `BCmodel.h5` |

---

## 🔧 Future Improvements

- [ ] Add transfer learning (ResNet50, InceptionV3) for higher accuracy
- [ ] Experiment with a 4th class (no tumor) for more realistic deployment
- [ ] Add GradCAM visualizations to highlight tumor regions in predictions
- [ ] Build a web interface using Streamlit or Gradio for demo
- [ ] Export to TFLite for edge/mobile deployment
- [ ] Add precision/recall/F1 scores per class in the README

---

## 📄 References

- [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Glioma — Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/glioma/symptoms-causes/syc-20350251)
- [Meningioma — American Brain Tumor Association](https://www.abta.org/tumor_types/meningioma/)

---

## 🙋 Author

**Mindbender66**
- 🐙 GitHub: [@Mindbender66](https://github.com/Mindbender66)

---

## ⚠️ Disclaimer

> This project is for **educational and research purposes only**. It is **not** a medical diagnostic tool and should **never** replace professional medical advice, diagnosis, or treatment.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
