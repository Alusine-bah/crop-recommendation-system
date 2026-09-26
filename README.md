# 🌾 Crop Recommendation System

A machine learning-based system that recommends the most suitable crop to grow based on soil nutrients and environmental conditions.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-App-yellow.svg)](https://gradio.app/)

---

## 📌 Project Overview

Choosing the right crop for a given plot of land is one of the most consequential decisions a farmer makes. Wrong choices waste water, fertilizer, and an entire growing season. This project trains and compares five machine learning classifiers on soil and climate data, then deploys the best-performing model as an interactive web app that returns a crop recommendation from seven input values.

**Input features:** Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, soil pH, rainfall
**Output:** One of 22 crop labels (rice, maize, chickpea, … coffee)

**Best model:** Random Forest Classifier — **99.32% test accuracy**

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle "Crop Recommendation Dataset" |
| File | `Crop_recommendation.csv` |
| Rows | 2,200 |
| Columns | 8 (7 features + 1 target) |
| Missing values | 0 |
| Target classes | 22 |
| Class balance | Perfectly balanced — 100 samples per crop |

### Feature ranges

| Feature | Min | Max | Type |
|---|---|---|---|
| N | 0 | 140 | int |
| P | 5 | 145 | int |
| K | 5 | 205 | int |
| temperature | 8.83 | 43.68 | float (°C) |
| humidity | 14.26 | 99.98 | float (%) |
| ph | 3.50 | 9.94 | float |
| rainfall | 20.21 | 298.56 | float (mm) |

The 22 crops are: rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee.

![Crop Distribution](figures/crop_distribution.png)

---

## 🧠 Approach

### 1. Preprocessing
- Loaded with Pandas; verified shape, dtypes, and absence of nulls.
- Separated features (`X`) from target (`y = label`).
- **80/20 train-test split** with `random_state=42` (1,760 train / 440 test).
- Applied `StandardScaler` — fit on the training set only, then applied to the test set to avoid data leakage.

### 2. Models trained
Five classifiers were trained and compared on the same split:

| Model | Test Accuracy |
|---|---|
| Logistic Regression (`max_iter=5000`) | 96.36% |
| K-Nearest Neighbors | 95.68% |
| Support Vector Machine (RBF) | 96.82% |
| Decision Tree | 98.86% |
| **Random Forest** | **99.32%** |

### 3. Final model
The **Random Forest Classifier** was selected for its top accuracy, stability, and strong generalization across all 22 classes. It was retrained on the full training set and used for all subsequent predictions.

### 4. Deployment
A Gradio interface wraps the model. The user enters seven numeric values; the app scales them with the saved `StandardScaler` and returns the predicted crop.

---

## 📈 Results

**Random Forest — 99.32% accuracy on the held-out test set (440 samples).**

The model correctly classified 437 of 440 test samples. Its advantage over the Decision Tree (98.86%) comes from bagging, which reduces the variance that a single tree is prone to.

### Example prediction

```python
predict_crop(N=90, P=42, K=43, temp=20.5, humidity=80, ph=6.5, rainfall=200)
# → 'rice'