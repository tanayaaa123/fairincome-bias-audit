"""
Bias/Fairness Audit of an Income Prediction Model
====================================================
Dataset: UCI Adult Income (predict whether income >50K)
Protected attributes examined: sex, race

Pipeline:
1. Train a baseline XGBoost/LogReg model (no fairness constraints)
2. Audit it: accuracy per group, demographic parity, equal opportunity
3. Show a proxy-variable leak (e.g. relationship/marital-status correlates with sex)
4. Apply a mitigation technique (threshold adjustment per group - post-processing)
5. Re-audit and report the accuracy-fairness tradeoff honestly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json

pd.set_option('display.width', 120)

# -----------------------------
# 1. LOAD & CLEAN DATA
# -----------------------------
COLS = [
    'age','workclass','fnlwgt','education','education_num','marital_status',
    'occupation','relationship','race','sex','capital_gain','capital_loss',
    'hours_per_week','native_country','income'
]
df = pd.read_csv('adult.csv', header=None, names=COLS, skipinitialspace=True)

# clean target
df['income'] = df['income'].str.strip().str.rstrip('.')
df['income'] = (df['income'] == '>50K').astype(int)

for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].str.strip()

df = df.replace('?', np.nan).dropna()

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Base rate (overall % earning >50K): {df['income'].mean():.3f}")
print(f"Base rate by sex:\n{df.groupby('sex')['income'].mean()}")
print(f"Base rate by race:\n{df.groupby('race')['income'].mean()}")

# -----------------------------
# 2. ENCODE FEATURES
# -----------------------------
df_model = df.copy()
cat_cols = ['workclass','education','marital_status','occupation',
            'relationship','race','sex','native_country']

encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df_model[c] = le.fit_transform(df_model[c])
    encoders[c] = le

feature_cols = [c for c in df_model.columns if c not in ['income']]
X = df_model[feature_cols]
y = df_model['income']

# keep original sex/race labels for group analysis after split
sex_labels = df['sex'].values
race_labels = df['race'].values

X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = train_test_split(
    X, y, sex_labels, race_labels, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 3. BASELINE MODEL (no fairness constraint)
# -----------------------------
clf = LogisticRegression(max_iter=1000, class_weight=None)
clf.fit(X_train_scaled, y_train)

y_prob = clf.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

overall_acc = accuracy_score(y_test, y_pred)
print(f"\n=== BASELINE MODEL ===")
print(f"Overall accuracy: {overall_acc:.3f}")

# -----------------------------
# 4. FAIRNESS AUDIT FUNCTIONS
# -----------------------------
def audit_group(y_true, y_pred, group_labels, group_name):
    """Compute accuracy, selection rate (demographic parity), and TPR (equal opportunity) per group."""
    results = {}
    for g in np.unique(group_labels):
        mask = group_labels == g
        yt, yp = y_true[mask], y_pred[mask]
        acc = accuracy_score(yt, yp)
        selection_rate = yp.mean()  # demographic parity metric
        # equal opportunity: TPR among actual positives
        pos_mask = yt == 1
        tpr = yp[pos_mask].mean() if pos_mask.sum() > 0 else np.nan
        results[g] = {'accuracy': round(acc, 3), 'selection_rate': round(selection_rate, 3), 'tpr': round(tpr, 3)}
    print(f"\n--- Fairness audit by {group_name} ---")
    for g, r in results.items():
        print(f"  {g:20s}  acc={r['accuracy']}  selection_rate={r['selection_rate']}  TPR={r['tpr']}")
    return results

y_test_arr = y_test.values
sex_audit = audit_group(y_test_arr, y_pred, sex_test, 'sex')
race_audit = audit_group(y_test_arr, y_pred, race_test, 'race')

# demographic parity gap = max selection rate - min selection rate across groups
sex_gap = max(v['selection_rate'] for v in sex_audit.values()) - min(v['selection_rate'] for v in sex_audit.values())
sex_tpr_gap = max(v['tpr'] for v in sex_audit.values()) - min(v['tpr'] for v in sex_audit.values())
print(f"\nDemographic parity gap (sex): {sex_gap:.3f}")
print(f"Equal opportunity gap (sex, TPR difference): {sex_tpr_gap:.3f}")


# 5. PROXY VARIABLE CHECK

# Does 'relationship' (e.g. Husband/Wife/Own-child) leak sex even though sex is a separate column?
print("\n=== PROXY VARIABLE CHECK ===")
crosstab = pd.crosstab(df['relationship'], df['sex'], normalize='index')
print(crosstab)
print(">> 'relationship' strongly predicts sex (e.g. 'Husband' vs 'Wife' categories almost")
print("   perfectly separate male/female). Even if we DROP the sex column, the model can")
print("   still discriminate by sex indirectly through 'relationship'. This is the proxy problem.")


# 6. MITIGATION: PER-GROUP THRESHOLD ADJUSTMENT (post-processing)

# Instead of one global threshold (0.5), pick per-group thresholds that
# equalize the selection rate across sex groups (demographic parity mitigation).
print("\n=== MITIGATION: Per-group threshold adjustment ===")

target_overall_rate = y_pred.mean()  # keep overall approval rate roughly similar

def find_threshold_for_rate(probs, target_rate):
    """Find the probability threshold that yields approx target selection rate."""
    sorted_probs = np.sort(probs)[::-1]
    n_select = int(target_rate * len(probs))
    if n_select == 0:
        return 1.01
    return sorted_probs[min(n_select, len(sorted_probs)-1)]

new_pred = np.zeros_like(y_pred)
thresholds = {}
for g in np.unique(sex_test):
    mask = sex_test == g
    thresh = find_threshold_for_rate(y_prob[mask], target_overall_rate)
    thresholds[g] = round(thresh, 3)
    new_pred[mask] = (y_prob[mask] >= thresh).astype(int)

print(f"Adjusted thresholds per group: {thresholds}")

mitigated_acc = accuracy_score(y_test, new_pred)
print(f"\nOverall accuracy BEFORE mitigation: {overall_acc:.3f}")
print(f"Overall accuracy AFTER mitigation:  {mitigated_acc:.3f}")
print(f"Accuracy cost of mitigation: {overall_acc - mitigated_acc:.3f}")

sex_audit_after = audit_group(y_test_arr, new_pred, sex_test, 'sex (AFTER mitigation)')
sex_gap_after = max(v['selection_rate'] for v in sex_audit_after.values()) - min(v['selection_rate'] for v in sex_audit_after.values())
print(f"\nDemographic parity gap BEFORE: {sex_gap:.3f}")
print(f"Demographic parity gap AFTER:  {sex_gap_after:.3f}")


# 7. SAVE RESULTS FOR DASHBOARD
results_summary = {
    'overall_acc_before': round(overall_acc, 4),
    'overall_acc_after': round(mitigated_acc, 4),
    'sex_audit_before': sex_audit,
    'sex_audit_after': sex_audit_after,
    'race_audit_before': race_audit,
    'demographic_parity_gap_before': round(sex_gap, 4),
    'demographic_parity_gap_after': round(sex_gap_after, 4),
    'equal_opportunity_gap_before': round(sex_tpr_gap, 4),
    'thresholds_after_mitigation': thresholds,
}

with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n\nSaved results_summary.json")
print("\n=== ONE-LINE TAKEAWAY FOR YOUR RESUME/INTERVIEW ===")
print(f"Baseline model had a {sex_gap:.1%} demographic parity gap by sex.")
print(f"Post-processing mitigation reduced it to {sex_gap_after:.1%},")
print(f"at a cost of {(overall_acc-mitigated_acc)*100:.1f} percentage points of accuracy.")
