# Fraud Detection Using Machine Learning with MLflow Experiment Tracking

**Author:** Abdelrahman Gamal (abdelrahman234013)

This project builds and evaluates various machine learning models on a real-world, imbalanced dataset for fraud detection. The dataset consists of both numerical and categorical features (nominal and ordinal variables), with a very small share of fraud cases within the total observations. This class imbalance required special consideration throughout preprocessing, model choice, and evaluation metrics.

---

## Data Preprocessing & Pipeline Design

- Studied the dataset structure by understanding its feature types and class distribution. The target variable, `FraudFound`, was highly imbalanced — fraud cases constituted about **6%** of the data.
- Built the preprocessing pipeline with `ColumnTransformer` to avoid data leakage and guarantee reproducibility:
  - `StandardScaler` for numerical features
  - `OneHotEncoder` for nominal categorical features
  - `OrdinalEncoder` (with explicit ordering) for ordinal categorical features
- Preprocessing was applied at the top level across all models to ensure features were transformed consistently throughout the experiments.

## Baseline Models

Implemented and compared several baseline classifiers:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier

The Decision Tree classifier was optimized using `GridSearchCV`, using **F1-score** as the optimization criterion since it's more suitable for imbalanced datasets.

**Insight:** Although some algorithms achieved high accuracy, the recall for the fraud class was severely low — a classic symptom of class imbalance.

## Feature Selection and Clustering Experiments

- Tested whether reducing the feature space could improve performance using:
  - `SelectKBest`
  - Model-based feature selection with Decision Tree, KNN, and SVM
- Also evaluated feature augmentation with clustering techniques as an optional approach.

**Insight:** Feature selection produced only a slight gain in accuracy, with no significant improvement in recall or F1-score.

## Handling Class Imbalance with SMOTE

- Applied SMOTE within an `imblearn` pipeline, positioned after preprocessing and before model training, to generate synthetic fraud samples only on the training set.
- Applying SMOTE to the Decision Tree classifier resulted in a slight decrease in overall accuracy, but clear improvements in **recall**, **F1-score**, and **ROC-AUC**.

**Insight:** SMOTE successfully reduced bias toward the majority class and improved the model's ability to detect fraudulent cases — the core objective of fraud detection.

## MLflow Experiment Tracking

- All experiments were tracked using MLflow for a systematic and reproducible process.
- Each model's training run was wrapped in an MLflow run context, logging:
  - Hyperparameters
  - Evaluation metrics
  - The trained pipeline
- Used tags to group experiments into stages: **baseline**, **feature selection**, and **imbalance handling**.
- The MLflow UI was run locally on `localhost:5000`.

## Conclusion

- Accuracy alone was found to be an inappropriate metric for this fraud detection task due to class imbalance.
- Both the baseline and feature-selection models showed high accuracy but failed to identify the fraud class effectively.
- Adding **SMOTE** improved recall and F1-score, making **Decision Tree + SMOTE** the best-performing model overall.
- MLflow proved highly effective for managing and tracking the experiments.

---

## Tech Stack
- Python
- scikit-learn (`ColumnTransformer`, `GridSearchCV`, `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder`)
- imbalanced-learn (SMOTE, imblearn pipeline)
- MLflow

## Course Context
Machine Learning course, Group G-9, project code 25CSAI03H.
