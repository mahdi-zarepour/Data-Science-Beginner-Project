# Adult Census Income Dataset

## Overview

This repository contains a complete end-to-end data science pipeline for the **Adult Census Income Dataset**. The project covers the entire lifecycle of a machine learning workflow, including:

- **Data Cleaning & Preprocessing** – Handling missing values, encoding target variable, and preparing the data.
- **Exploratory Data Analysis (EDA)** – Visualizing distributions, correlations, and patterns to gain insights into the data.
- **Model Building & Evaluation** – Training multiple classification models, tuning hyperparameters, and evaluating performance using appropriate metrics.

The goal is to predict whether an individual's annual income exceeds **$50,000** based on demographic and employment-related attributes.

## 📁 Dataset

The dataset used is the classic **Adult Census Income** dataset, originally from the UCI Machine Learning Repository. It contains demographic and employment-related features used to predict whether an individual's income exceeds $50K per year.


## ⚙️ Preprocessing Notebook:

The notebook performs the following preprocessing tasks:

1. **Load the Data**  
   The dataset is loaded from a CSV file using `pandas`.

2. **Drop Unnecessary Columns**  
   - `Unnamed: 0` (index column)  
   - `education-num` (redundant with `education`)

3. **Encode Target Variable**  
   - The `target` column is converted to binary format:  
     - `>50K` → `1`  
     - `<=50K` → `0`

4. **Handle Missing Values**  
   - Missing values in `workclass`, `occupation`, and `native-country` are identified.  
   - The `?` placeholder is replaced with `NaN`.  
   - Rows with any missing values are dropped, resulting in a clean dataset.

5. **Save Cleaned Data**  
   - The cleaned DataFrame is saved as `adult_cleaned.csv`.

## Requirements

To run the notebook, you'll need the following Python libraries:

- pandas
- numpy

You can install them using pip:

```bash
pip install pandas numpy
```

## 📊 EDA Notebook

The primary goal is to understand the data's structure, identify patterns, and uncover relationships between features and the target variable (`income`), where the target indicates whether an individual's income exceeds $50K/year.

The original EDA is performed using Python with libraries such as `pandas`, `matplotlib`, and `seaborn`.

---

## 📁 Dataset Overview

The dataset contains 45,222 rows and 14 columns. It is often used for classification tasks to predict if a person's annual income exceeds $50,000 based on census data.

- **Target Variable:** `target` (Binary: 0 = <=50K, 1 = >50K)
- **Features:** A mix of categorical and numerical variables, including:
  - `age`
  - `workclass`
  - `fnlwgt` (final weight)
  - `education`
  - `marital-status`
  - `occupation`
  - `relationship`
  - `race`
  - `sex`
  - `capital-gain`
  - `capital-loss`
  - `hours-per-week`
  - `native-country`

## 📊 Key Steps in the EDA

### 1. Data Loading and Initial Exploration
The dataset is loaded into a Pandas DataFrame, and the first few rows are displayed to understand its structure.

### 2. Data Profiling
- **`df.info()`:** Provides a concise summary of the DataFrame, including column data types, non-null counts, and memory usage. The dataset has no missing values.
- **`df.describe()`:** Generates descriptive statistics for numerical features, revealing insights into central tendencies and distributions. For example, the `capital-gain` feature has a very high standard deviation due to a few large values.
- **`df.describe(include='object')`:** Provides statistics for categorical features, including their unique counts, most frequent values, and frequencies. For instance, "United-States" is the most common `native-country`.

### 3. Correlation Analysis
A correlation matrix was created for the numerical features to examine linear relationships between them. A heatmap visualizes these correlations.

The heatmap shows that while some correlations exist, they are generally not very strong. For example, `age` and `hours-per-week` show a weak positive correlation.

### 4. Visualizing Relationships
A `pairplot` is used to visualize pairwise relationships between numerical features and the `target` variable. This helps identify potential separability between the two income classes (`target=0` vs `target=1`).


### 5. Categorical Feature Distribution
Bar plots are generated for all categorical features to understand the frequency of their categories. This step helps identify major categories and potential data imbalances.


## 📈 Key Findings

*   **Imbalanced Target:** The `target` variable is imbalanced, with significantly more instances of `income <=50K` (0) than `income >50K` (1). This is a crucial consideration for model building and evaluation.

*   **Dominant Categories:**
    *   `workclass`: `Private`
    *   `native-country`: `United-States`
    *   `marital-status`: `Married-civ-spouse`
    *   `sex`: `Male`
*   **Weak Correlations:** There are no strong linear correlations between numerical features.

## 🛠️ Requirements

To run the notebook, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib

You can install them using pip:

```bash
pip install pandas numpy matplotlib
```

## Model Notebook

This notebook implements and compares multiple machine learning models to predict whether an individual's income exceeds $50K/year using the UCI Adult Census dataset.

## 📁 Dataset

The dataset used is `adult_cleaned.csv`, which contains demographic and employment information of individuals. The target variable is `target` (binary: 0 = ≤50K, 1 = >50K).

## 📊 Features

### Numerical Features
- age
- fnlwgt
- capital-gain
- capital-loss
- hours-per-week

### Categorical Features
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

## 🧠 Models Evaluated

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Tree-based classifier with hyperparameter tuning |
| K-Nearest Neighbors | Distance-based classifier |
| Naive Bayes | Probabilistic classifier (Gaussian) |
| Voting Classifier (Hard) | Ensemble of top models |
| Neural Network (ANN) | Dense multilayer perceptron with dropout and Eaely Stopping|

## ⚙️ Preprocessing Pipeline

- **Numerical**: Median imputation + StandardScaler
- **Categorical**: Mode imputation + One-Hot Encoding
- **Split**: 80/20 train-test with stratification
- **Validation**: 5-fold Stratified Cross-Validation

## 🔍 Hyperparameter Tuning

GridSearchCV was used for the following models:

- **Decision Tree**: `max_depth` [10, 20, None], `criterion` ['gini', 'entropy']
- **KNN**: `n_neighbors` [3, 5, 10, 15]
- **Logistic Regression & Naive Bayes**: Default parameters

## 📈 Results

| Model | CV Accuracy | Test Accuracy |
|-------|-------------|---------------|
| **Decision Tree** | **0.8528** | **0.8558** |
| Logistic Regression | 0.8445 | 0.8495 |
| KNN | 0.8432 | 0.8428 |
| Naive Bayes | 0.4543 | 0.4783 |
| Voting Classifier (Hard) | — | 0.8547 |
| Neural Network | — | 0.8528 |


## 🛠️ Requirements

To run the notebook, you'll need the following Python libraries:

- pandas
- numpy
- scikit-learn
- tensorflow

You can install them using pip:

```bash
pip install pandas numpy scikit-learn tensorflow
```