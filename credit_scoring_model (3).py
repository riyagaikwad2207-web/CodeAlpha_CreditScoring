# ============================================================
# CREDIT SCORING MODEL — CodeAlpha Internship Task 1
# ============================================================
# Objective: Predict creditworthiness using past financial data
# Algorithms: Logistic Regression, Decision Tree, Random Forest
# Metrics: Precision, Recall, F1-Score, ROC-AUC
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score
)

# ─────────────────────────────────────────────
# STEP 1: Generate Realistic Synthetic Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("  CREDIT SCORING MODEL — CodeAlpha Internship Task 1")
print("=" * 60)
print("\n[1/6] Generating dataset...")

np.random.seed(42)
n = 2000

age            = np.random.randint(21, 70, n)
income         = np.random.normal(55000, 20000, n).clip(15000, 150000)
debt_ratio     = np.random.beta(2, 5, n)
num_loans      = np.random.randint(0, 10, n)
missed_payments= np.random.randint(0, 6, n)
credit_history = np.random.randint(1, 20, n)
employment_yrs = np.random.randint(0, 30, n)
savings        = np.random.normal(20000, 15000, n).clip(0, 100000)

# Target: 1 = defaulted (bad credit), 0 = good credit
default_score = (
    -0.3 * (income / 150000)
    + 0.4 * debt_ratio
    + 0.25 * (missed_payments / 5)
    + 0.2  * (num_loans / 10)
    - 0.15 * (credit_history / 20)
    - 0.1  * (employment_yrs / 30)
    - 0.1  * (savings / 100000)
    + np.random.normal(0, 0.1, n)
)
threshold = np.percentile(default_score, 72)
target = (default_score > threshold).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income.astype(int),
    'debt_ratio': debt_ratio.round(4),
    'num_loans': num_loans,
    'missed_payments': missed_payments,
    'credit_history_years': credit_history,
    'employment_years': employment_yrs,
    'savings': savings.astype(int),
    'defaulted': target
})

df.to_csv('credit_data.csv', index=False)
print(f"   ✔ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   ✔ Default rate: {target.mean()*100:.1f}%  |  Good credit: {(1-target).mean()*100:.1f}%")

# ─────────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────────
print("\n[2/6] Feature engineering...")

df['debt_to_income']    = (df['debt_ratio'] * df['income']).astype(int)
df['savings_per_loan']  = (df['savings'] / (df['num_loans'] + 1)).round(2)
df['payment_risk']      = (df['missed_payments'] / (df['credit_history_years'] + 1)).round(4)

features = [
    'age', 'income', 'debt_ratio', 'num_loans', 'missed_payments',
    'credit_history_years', 'employment_years', 'savings',
    'debt_to_income', 'savings_per_loan', 'payment_risk'
]

X = df[features]
y = df['defaulted']
print(f"   ✔ Features used: {len(features)}")

# ─────────────────────────────────────────────
# STEP 3: Train / Test Split & Scaling
# ─────────────────────────────────────────────
print("\n[3/6] Splitting and scaling data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"   ✔ Training set: {X_train.shape[0]} samples")
print(f"   ✔ Test set:     {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# STEP 4: Train 3 Models
# ─────────────────────────────────────────────
print("\n[4/6] Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_proba),
        'cm':        confusion_matrix(y_test, y_pred)
    }
    print(f"   ✔ {name} trained  |  ROC-AUC: {results[name]['roc_auc']:.4f}")

# ─────────────────────────────────────────────
# STEP 5: Print Full Metrics Report
# ─────────────────────────────────────────────
print("\n[5/6] Generating metrics report...")
print("\n" + "─" * 60)
print(f"{'Model':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}")
print("─" * 60)
for name, r in results.items():
    print(f"{name:<25} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>9.4f}")
print("─" * 60)

best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
print(f"\n🏆 Best Model: {best_model_name}  (ROC-AUC = {results[best_model_name]['roc_auc']:.4f})")

# ─────────────────────────────────────────────
# STEP 6: Visualizations (saved to PNG)
# ─────────────────────────────────────────────
print("\n[6/6] Creating visualizations...")

palette = {"good": "#2ecc71", "bad": "#e74c3c"}
colors  = ['#3498db', '#e67e22', '#9b59b6']

# ── Figure 1: EDA ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Credit Scoring — Exploratory Data Analysis', fontsize=16, fontweight='bold', y=1.01)

# 1a. Target distribution
counts = y.value_counts()
axes[0,0].bar(['Good Credit', 'Defaulted'], counts.values, color=['#2ecc71','#e74c3c'], edgecolor='white', linewidth=1.5)
axes[0,0].set_title('Target Distribution', fontweight='bold')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0,0].text(i, v + 20, str(v), ha='center', fontweight='bold')

# 1b. Income distribution
axes[0,1].hist(df[df['defaulted']==0]['income'], bins=30, alpha=0.6, color='#2ecc71', label='Good Credit')
axes[0,1].hist(df[df['defaulted']==1]['income'], bins=30, alpha=0.6, color='#e74c3c', label='Defaulted')
axes[0,1].set_title('Income Distribution', fontweight='bold')
axes[0,1].set_xlabel('Income ($)')
axes[0,1].legend()

# 1c. Missed payments
axes[0,2].hist(df[df['defaulted']==0]['missed_payments'], bins=6, alpha=0.6, color='#2ecc71', label='Good Credit')
axes[0,2].hist(df[df['defaulted']==1]['missed_payments'], bins=6, alpha=0.6, color='#e74c3c', label='Defaulted')
axes[0,2].set_title('Missed Payments Distribution', fontweight='bold')
axes[0,2].set_xlabel('Missed Payments')
axes[0,2].legend()

# 1d. Debt ratio boxplot
df_plot = df.copy()
df_plot['Credit Status'] = df_plot['defaulted'].map({0:'Good Credit', 1:'Defaulted'})
axes[1,0].boxplot([df[df['defaulted']==0]['debt_ratio'], df[df['defaulted']==1]['debt_ratio']],
                   labels=['Good Credit','Defaulted'],
                   patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.6))
axes[1,0].set_title('Debt Ratio by Credit Status', fontweight='bold')
axes[1,0].set_ylabel('Debt Ratio')

# 1e. Correlation heatmap
corr = df[features + ['defaulted']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr[['defaulted']].drop('defaulted'), ax=axes[1,1],
            annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            linewidths=0.5, cbar=False)
axes[1,1].set_title('Feature Correlation with Default', fontweight='bold')

# 1f. Age vs Income scatter
sc = axes[1,2].scatter(df['age'], df['income'], c=df['defaulted'],
                        cmap='RdYlGn_r', alpha=0.4, s=15)
axes[1,2].set_title('Age vs Income', fontweight='bold')
axes[1,2].set_xlabel('Age')
axes[1,2].set_ylabel('Income ($)')
plt.colorbar(sc, ax=axes[1,2], label='Defaulted')

plt.tight_layout()
plt.savefig('01_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✔ Saved: 01_eda_analysis.png")

# ── Figure 2: Model Performance ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Performance — Confusion Matrices', fontsize=15, fontweight='bold')

for i, (name, r) in enumerate(results.items()):
    cm = r['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Good','Default'], yticklabels=['Good','Default'],
                linewidths=1, linecolor='white', annot_kws={'size':14,'weight':'bold'})
    axes[i].set_title(f"{name}\nROC-AUC: {r['roc_auc']:.4f}", fontweight='bold')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('02_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✔ Saved: 02_confusion_matrices.png")

# ── Figure 3: ROC Curves ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC Curves & Metrics Comparison', fontsize=14, fontweight='bold')

for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    axes[0].plot(fpr, tpr, color=colors[i], lw=2,
                 label=f"{name} (AUC={r['roc_auc']:.3f})")

axes[0].plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
axes[0].fill_between([0,1],[0,1], alpha=0.05, color='grey')
axes[0].set_title('ROC Curves', fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Metrics bar chart
metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metric_names))
width = 0.25

for i, (name, r) in enumerate(results.items()):
    vals = [r['precision'], r['recall'], r['f1'], r['roc_auc']]
    bars = axes[1].bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85, edgecolor='white')

axes[1].set_title('Metrics Comparison', fontweight='bold')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metric_names)
axes[1].set_ylim(0, 1.1)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.4, label='0.8 threshold')

plt.tight_layout()
plt.savefig('03_roc_and_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✔ Saved: 03_roc_and_metrics.png")

# ── Figure 4: Feature Importance (Random Forest) ────────────
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importances.index, importances.values,
               color=plt.cm.viridis(np.linspace(0.2, 0.9, len(importances))),
               edgecolor='white', linewidth=0.5)
ax.set_title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('04_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✔ Saved: 04_feature_importance.png")

# ─────────────────────────────────────────────
# PREDICTION DEMO
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("  SAMPLE PREDICTION DEMO")
print("─" * 60)

sample = pd.DataFrame([{
    'age': 35,
    'income': 45000,
    'debt_ratio': 0.55,
    'num_loans': 4,
    'missed_payments': 2,
    'credit_history_years': 8,
    'employment_years': 5,
    'savings': 8000,
    'debt_to_income': int(0.55 * 45000),
    'savings_per_loan': round(8000 / 5, 2),
    'payment_risk': round(2 / 9, 4)
}])

sample_sc = scaler.transform(sample)
best = results[best_model_name]['model']
pred  = best.predict(sample_sc)[0]
proba = best.predict_proba(sample_sc)[0][1]

print(f"  Input: Age=35, Income=₹45,000, Debt Ratio=0.55, Missed Payments=2")
print(f"  Model: {best_model_name}")
print(f"  Default Probability: {proba*100:.1f}%")
print(f"  Prediction: {'⚠️  HIGH RISK — Likely to Default' if pred==1 else '✅  LOW RISK — Creditworthy'}")
print("─" * 60)
print("\n✅ All done! Files saved:")
print("   credit_data.csv, 01_eda_analysis.png,")
print("   02_confusion_matrices.png, 03_roc_and_metrics.png,")
print("   04_feature_importance.png")
