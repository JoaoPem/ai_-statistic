# Artificial Intelligence Project with Statistics

This project was developed as part of a university course, involving the use of **Artificial Intelligence techniques based on statistics** to solve **regression** and **classification** problems, using **supervised models** and validation through **Monte Carlo simulations**.

## 📚 Libraries Used

- `numpy`
- `matplotlib`
- `seaborn`

---

## 🔍 Introduction

The project consists of two main stages:

1. **Regression Task**: Development of a model capable of making **quantitative predictions** based on continuous variables.
2. **Classification Task**: Creation of a model to make **qualitative predictions**, classifying samples into different categories.

Both models use supervised learning techniques through cost function (loss function) minimization.

---

## 📈 Regression Task

### 🔹 Objective
Predict the level of **enzymatic activity** based on the **temperature** and **pH** of the solution.

### 🔹 Steps Taken

1. **Initial Visualization**: Scatter plots between independent and dependent variables.
2. **Pre-processing**: Organizing data into matrices `X` (independent variables) and `y` (dependent variable).
3. **Implemented Models**:
   - Traditional OLS (Ordinary Least Squares)
   - Regularized OLS (Tikhonov) with `λ = [0.0, 0.25, 0.5, 0.75, 1.0]`
   - Mean of observed values
4. **Monte Carlo Validation**:
   - 500 simulations with an 80/20 train/test split.
   - Evaluation metric: **Residual Sum of Squares (RSS)**
5. **Results Analysis**:
   - Calculation of mean, standard deviation, maximum, and minimum RSS for each model.
   - Results presented in **tables and graphs**.

---

## 🤖 Classification Task

### 🔹 Objective
Classify facial expressions based on **electromyography (EMG)** signals obtained from sensors placed on facial muscles.

### 🔹 Dataset
- File: `EMGsDataset.csv`
- Features:
  - N = 50000 samples
  - p = 2 sensors (Corrugator Supercilii and Zygomaticus Major)
  - C = 5 classes (Neutral, Smile, Raised Eyebrows, Surprised, Grumpy)

### 🔹 Steps Taken

1. **Data Organization**:
   - `X`: Feature matrix
   - `Y`: Category matrix
2. **Initial Visualization**: Scatter plot distinguishing between classes.
3. **Implemented Models**:
   - Traditional OLS
   - Traditional Gaussian Classifier
   - Gaussian with Training Set Covariance
   - Gaussian with Aggregated Covariance Matrix
   - Naive Bayes Classifier
   - Regularized Gaussian (Friedman) with `λ = [0.25, 0.5, 0.75, 1.0]`
4. **Monte Carlo Validation**:
   - 500 simulations with an 80/20 train/test split.
   - Evaluation metric: **Accuracy**
5. **Results Analysis**:
   - Calculation of mean, standard deviation, maximum, and minimum accuracy for each model.

---
