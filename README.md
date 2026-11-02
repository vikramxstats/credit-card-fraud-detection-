# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions in a highly imbalanced dataset, using Logistic Regression, Decision Tree and Random Forest, with SMOTE applied to the training split only.

## Problem Statement

Fraudulent transactions are rare but costly. In this dataset only 0.17% of transactions are fraud, so accuracy is misleading: a model that predicts every transaction as normal is 99.83% accurate and catches no fraud. The goal is to detect fraud with high precision and recall while keeping false alarms low.

## Dataset

- Source: [Credit Card Fraud Detection dataset (ULB / Worldline) on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions by European cardholders over two days in September 2013
- 284,807 transactions, 492 frauds (0.17%)
- 1,081 exact duplicate rows removed, leaving 283,726 rows and 473 frauds
- No missing values

| Column | Description |
|---|---|
| Time | Seconds elapsed since the first transaction in the dataset |
| V1 to V28 | Principal components from PCA (original features are anonymised for confidentiality) |
| Amount | Transaction amount |
| Class | Target: 0 = normal, 1 = fraud |

The dataset file `creditcard.csv` is not included in this repository because of its size. Download it from the Kaggle link above and place it next to the notebook.

## Approach

1. Exploratory data analysis: class distribution, null checks, Amount and Time behaviour by class, correlation with the target
2. Removed 1,081 duplicate rows
3. Stratified 80/20 train-test split (`random_state=42`): 226,980 training rows (378 frauds) and 56,746 test rows (95 frauds)
4. Standardised features with `StandardScaler`, fitted on the training data only
5. Applied SMOTE on the training split only, to avoid data leakage (226,602 normal and 226,602 fraud rows after resampling)
6. Trained and compared Logistic Regression, Decision Tree (`max_depth=8`) and Random Forest (100 trees)
7. Evaluated with confusion matrix, precision, recall, F1, ROC-AUC and PR-AUC
8. Analysed decision thresholds from 0.10 to 0.90
9. Saved the final Random Forest with joblib

## Results

Test set: 56,746 transactions, 95 of them fraud.

| Model | Precision | Recall | False Positives | False Negatives |
|---|---|---|---|---|
| Logistic Regression | 0.05 | 0.87 | 1,482 | 12 |
| Decision Tree | 0.08 | 0.80 | 880 | 19 |
| Random Forest | 0.91 | 0.77 | 7 | 22 |

Best model: Random Forest

| Metric | Value |
|---|---|
| Precision | 91.25% |
| Recall | 76.84% |
| F1-Score | 83.43% |
| ROC-AUC | 0.964 |
| PR-AUC | 0.812 |
| Frauds detected | 73 of 95 |
| False positives | 7 |

Confusion matrix (Random Forest):

|  | Predicted Normal | Predicted Fraud |
|---|---|---|
| Actual Normal | 56,644 | 7 |
| Actual Fraud | 22 | 73 |

## Threshold Analysis

Lowering the threshold catches more fraud but raises false alarms.

| Threshold | Precision | Recall |
|---|---|---|
| 0.10 | 0.33 | 0.84 |
| 0.30 | 0.80 | 0.81 |
| 0.50 | 0.91 | 0.77 |
| 0.90 | 0.98 | 0.59 |

A threshold of 0.30 gives about 81% recall at 80% precision.

## Top Features (Random Forest)

| Feature | Importance |
|---|---|
| V14 | 0.192 |
| V10 | 0.119 |
| V12 | 0.101 |
| V17 | 0.093 |
| V4 | 0.089 |

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn, Joblib, Jupyter Notebook

## How to Run

```bash
git clone https://github.com/vikramxstats/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib jupyter
jupyter notebook
```

Download `creditcard.csv` from Kaggle, place it in the project folder, open the notebook and run all cells.

## Limitations and Future Work

- The decision threshold was analysed on the test set; a separate validation set or cross-validation would be more rigorous
- No hyperparameter tuning or cross-validation was performed
- The test set contains only 95 frauds, so metrics are noisy
- Possible improvements: XGBoost or LightGBM, `class_weight` as an alternative to SMOTE, a scikit-learn pipeline that includes the scaler, and calibrated probabilities

## Author

Vikram Singh
MSc Statistics & Computing, Banaras Hindu University
[LinkedIn](https://linkedin.com/in/vikram-singh-dst27) | [GitHub](https://github.com/vikramxstats)
