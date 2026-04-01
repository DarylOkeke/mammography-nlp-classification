"""
SPR 2026 Mammography Report Classification
Baseline Pipeline: TF-IDF + LinearSVC (word + char ngrams)

Architecture decisions (see spr_notes.md for reasoning):
- Portuguese text → char ngrams capture morphology well
- LinearSVC outperforms LR dramatically on this task (~0.71 vs ~0.49 dedup-aware macro F1)
- Dedup-aware stratified CV gives honest estimate
- Balanced class weights compensate for extreme BI-RADS 2 dominance (87.4%)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
train = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/train.csv")
test  = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/test.csv")
sub   = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/submission.csv")

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Target distribution:\n{train.target.value_counts().sort_index()}")

# ─── 2. DEDUPLICATION ────────────────────────────────────────────────────────
# 54% of training rows are exact duplicates of other rows.
# For CV, we must split on unique texts to avoid cross-fold leakage.
# For training the final model, we use ALL rows (more signal for rare classes).

dedup = (
    train
    .groupby("report")
    .agg(target=("target", lambda x: x.value_counts().index[0]))
    .reset_index()
)
print(f"\nUnique report texts: {len(dedup)} (from {len(train)} rows)")

X_cv   = dedup["report"].values
y_cv   = dedup["target"].values
X_full = train["report"].fillna("").values
y_full = train["target"].values
X_test = test["report"].fillna("").values

# ─── 3. FEATURE BUILDERS ─────────────────────────────────────────────────────
def build_features(X_tr, X_val=None):
    """
    Combined word (1-3gram) + char_wb (3-5gram) TF-IDF features.
    Returns (X_tr_feats, X_val_feats) or just X_tr_feats if X_val is None.
    """
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3),
        min_df=2, max_features=50000, sublinear_tf=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=3, max_features=50000, sublinear_tf=True
    )
    X_tr_w = word.fit_transform(X_tr)
    X_tr_c = char.fit_transform(X_tr)
    X_tr_f = hstack([X_tr_w, X_tr_c])

    if X_val is not None:
        X_val_f = hstack([word.transform(X_val), char.transform(X_val)])
        return X_tr_f, X_val_f, word, char
    return X_tr_f, word, char


# ─── 4. DEDUP-AWARE CROSS-VALIDATION ─────────────────────────────────────────
CLASSES = list(range(7))
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(X_cv), dtype=int)
oof_probas = np.zeros((len(X_cv), 7))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
    X_tr_f, X_val_f, _, _ = build_features(X_cv[tr_idx], X_cv[val_idx])

    clf = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, class_weight="balanced", C=0.3), cv=3
    )
    clf.fit(X_tr_f, y_cv[tr_idx])
    oof_preds[val_idx]  = clf.predict(X_val_f)
    oof_probas[val_idx] = clf.predict_proba(X_val_f)

cv_f1 = f1_score(y_cv, oof_preds, average="macro")
print(f"\n=== CV Macro F1 (dedup-aware): {cv_f1:.4f} ===")
print(classification_report(y_cv, oof_preds, digits=3))

# Per-class breakdown for analysis
per_class = f1_score(y_cv, oof_preds, average=None, labels=CLASSES)
for c, f in zip(CLASSES, per_class):
    print(f"  BI-RADS {c}: F1={f:.3f}")

# ─── 5. CONFUSION MATRIX ─────────────────────────────────────────────────────
cm = confusion_matrix(y_cv, oof_preds, labels=CLASSES)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

# ─── 6. TRAIN FINAL MODEL ON ALL DATA ────────────────────────────────────────
# Train on full (duplicated) training set for maximum signal on rare classes.
print("\nTraining final model on full dataset...")
X_full_f, word_final, char_final = build_features(X_full)
test_f = hstack([word_final.transform(X_test), char_final.transform(X_test)])

final_clf = CalibratedClassifierCV(
    LinearSVC(max_iter=2000, class_weight="balanced", C=0.3), cv=3
)
final_clf.fit(X_full_f, y_full)

# ─── 7. GENERATE PREDICTIONS ─────────────────────────────────────────────────
test_probas = final_clf.predict_proba(test_f)
test_labels = final_clf.predict(test_f)

print("\nTest predictions:")
label_name = {
    0:"Incomplete", 1:"Negative", 2:"Benign",
    3:"ProbBenign", 4:"Suspicious", 5:"HighlySusp", 6:"KnownMalig"
}
for i, row in test.iterrows():
    p = test_labels[i]
    print(f"  {row.ID}: {p} ({label_name[p]}) | conf={test_probas[i].max():.3f}")

# ─── 8. BUILD SUBMISSION ─────────────────────────────────────────────────────
# test.csv may only contain the public portion (4 rows) of the full test set.
# For the remaining IDs in submission.csv not in test.csv, default to BI-RADS 2
# (majority class) as a safe placeholder — update when you have their text.

pred_map = dict(zip(test["ID"].values, test_labels))
sub["target"] = sub["ID"].map(pred_map).fillna(2).astype(int)

print("\nFinal submission:")
print(sub.to_string(index=False))

sub.to_csv("submission.csv", index=False)
print("\nSaved: submission.csv")
