# 💳 Credit Scoring Model — CodeAlpha Internship Task 1

> Predicting individual creditworthiness using Machine Learning classification algorithms.

---

## 📌 Objective

Build a machine learning model that predicts whether a person is likely to **default on a loan** based on their financial history and personal data.

---

## 🧠 Approach

Three classification algorithms were trained and compared:

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.7843 | 0.7143 | 0.7477 | **0.9370** ✅ |
| Decision Tree | 0.6667 | 0.6786 | 0.6726 | 0.8232 |
| Random Forest | 0.7692 | 0.6250 | 0.6897 | 0.9054 |

🏆 **Best Model: Logistic Regression** with ROC-AUC = 0.9370

---

## 📊 Key Features Used

| Feature | Description |
|---------|-------------|
| `age` | Age of the individual |
| `income` | Annual income |
| `debt_ratio` | Debt as a proportion of income |
| `num_loans` | Number of active loans |
| `missed_payments` | Number of missed payments |
| `credit_history_years` | Years of credit history |
| `employment_years` | Years of employment |
| `savings` | Total savings amount |
| `debt_to_income` | Engineered: Debt × Income |
| `savings_per_loan` | Engineered: Savings / Loans |
| `payment_risk` | Engineered: Missed Payments / History |

---

## 📁 Project Structure

```
CodeAlpha_CreditScoring/
│
├── credit_scoring_model.py     # Main Python script
├── credit_data.csv             # Generated dataset (2000 samples)
├── 01_eda_analysis.png         # Exploratory Data Analysis charts
├── 02_confusion_matrices.png   # Confusion matrices for all 3 models
├── 03_roc_and_metrics.png      # ROC curves & metrics bar chart
├── 04_feature_importance.png   # Random Forest feature importances
└── README.md                   # This file
```

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3. Run the model
```bash
python credit_scoring_model.py
```

---

## 📈 Visualizations

The script automatically generates 4 visualization files:

- **01_eda_analysis.png** — Target distribution, income, missed payments, debt ratio, correlation heatmap
- **02_confusion_matrices.png** — Confusion matrices for all 3 models
- **03_roc_and_metrics.png** — ROC curves overlay + Precision/Recall/F1/AUC bar chart
- **04_feature_importance.png** — Top features by importance (Random Forest)

---

## 🔍 Sample Prediction

```
Input: Age=35, Income=45000, Debt Ratio=0.55, Missed Payments=2
Model: Logistic Regression
Default Probability: 69.2%
Result: ⚠️ HIGH RISK — Likely to Default
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **scikit-learn** — ML models & metrics
- **matplotlib / seaborn** — Visualizations

---

## 👤 Author

**[Your Name]**
CodeAlpha Machine Learning Intern

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/YOUR_USERNAME)

---

*Submitted as part of CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model*
