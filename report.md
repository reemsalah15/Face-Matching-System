# ✅ **Face Age Prediction & Cross-Age Face Matching Report**

## **1. Objective**

The objective of this project is to develop a robust **face-matching system** that accurately identifies whether two images belong to the same individual despite differences in age. Additionally, the system predicts the estimated age for each image.

---

## **2. Dataset Selection & Preprocessing**

### **2.1 Dataset Choice**

We selected the **FG-NET Aging Dataset** from [Kaggle](https://www.kaggle.com/datasets/aiolapo/fgnet-dataset) because:

* It contains **images of individuals at different ages**, making it suitable for cross-age face recognition tasks.
* Images are labeled with age information embedded in the filenames.
* It is widely used in academic research for **age-invariant face recognition**.

The dataset consists of **1,002 images** of 82 individuals, aged between 0 and 69.

---

### **2.2 Preprocessing**

* **Filename Parsing**: Extracted `person_id` and `age` from filenames using regular expressions.
* **Face Cropping**: Detected and cropped faces using **MTCNN** to ensure only facial regions are used.
* **Image Resizing**: Resized to `224x224` for age prediction (ResNet50) and `112x112` for face matching (ArcFace/Facenet).
* **Age Binning**: Divided ages into bins to ensure stratified train-test split.

---

### **2.3 Data Augmentation**

To reduce overfitting, we applied:

* **Rotation** (±30 degrees)
* **Width/Height shift** (0.2)
* **Zoom** (0.2)
* **Brightness adjustment** (`0.7–1.3`)
* **Horizontal flip**

This augmentation improves generalization, especially since the dataset is relatively small.

---

## **3. Model Development**

### **3.1 Age Prediction Model**

#### **Architecture**

We used **Transfer Learning with ResNet50**:

* **Base Model**: Pre-trained on ImageNet (`include_top=False`).
* **Custom Head**:

  * Global Average Pooling
  * Dense(256, activation=‘relu’) + Dropout(0.3)
  * Dense(1, activation=‘linear’) for regression.

#### **Training Details**

* **Loss**: Mean Squared Error (MSE) → suitable for regression tasks.
* **Metric**: Mean Absolute Error (MAE) → easier to interpret (years).
* **Optimizer**: Adam (`lr=1e-4`)
* **Epochs**: 30
* **Batch size**: 16

#### **Performance**

* **Validation MAE**: \~4 years
* **Learning Curve** shows no significant overfitting after augmentation.

---

### **3.2 Face Matching Model**

We used a **pre-trained embedding model** instead of training from scratch:

* **Model**: **Facenet512** (via DeepFace)
* **Why Facenet512?**

  * Proven high accuracy in face verification tasks.
  * Outputs a 512-d embedding vector, robust to age variations.

#### **Matching Method**

1. Extract embeddings for both images.
2. Compute **L2 distance** (Euclidean) or **Cosine similarity**.
3. Classification rule:

   * **Same person** if distance ≤ **0.4** (tuned via ROC curve).

#### **Performance**

* **Accuracy**: \~76% on test pairs
* **ROC-AUC**: \~0.9

---

### **3.3 Full Pipeline**

The pipeline integrates both models:

1. Input two face images.
2. Predict ages (`predict_age()`).
3. Compare embeddings (`face_matching_inference()`):

   * Output: Predicted ages, distance, and same-person classification.

Example Output:

```
Image1 Age: 25.4 | Image2 Age: 31.2
Same Person? True (Distance=0.318)
```

---

## **4. System Capabilities**

### ✅ **Strengths**

* Works well with age gaps (childhood vs adulthood).
* Achieves high accuracy despite a small dataset.
* Data augmentation significantly reduced overfitting.
* Easy to extend to other datasets.

### ⚠ **Drawbacks**

* Limited by dataset size → may misclassify in extreme age gaps or poor-quality images.
* Age prediction MAE (\~4 years) may not be precise for very young ages.
* Performance depends on face detection quality (MTCNN may fail on low-resolution images).

---

## **5. Evaluation Metrics**

| **Task**       | **Metric** | **Score** |
| -------------- | ---------- | --------- |
| Age Prediction | MAE        | \~4 years |
| Face Matching  | Accuracy   | \~76%     |
| Face Matching  | ROC-AUC    | \~0.9     |

---

## **6. How to Run the System**

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/face-age-matching.git
   cd face-age-matching
   pip install -r requirements.txt
   ```

2. Run the notebook:
   [**face\_age\_matching.ipynb**](Notebook/face_matching_system_V2.ipynb)

3. Inference example:

   ```python
   img1 = "path_to_image1.jpg"
   img2 = "path_to_image2.jpg"
   full_inference(img1, img2)
   ```

---

## **7. Conclusion**

This project demonstrates that combining **transfer learning for age prediction** and **pre-trained face embedding models** (Facenet512) provides a reliable solution for **cross-age face verification**.
While improvements can be achieved with larger datasets and advanced age-invariant models (e.g., ArcFace or CARC), this system achieves competitive results with minimal computational resources.

---
