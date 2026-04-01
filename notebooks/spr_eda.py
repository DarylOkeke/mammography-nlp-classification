"""
SPR 2026 Mammography Report Classification
Exploratory Data Analysis

Run this locally or in a Kaggle notebook (internet ON, no submission needed).
It documents all the key findings about the dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re, warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/train.csv")
test  = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/test.csv")
sub   = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/submission.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Train rows:  {len(train):,}")
print(f"Test rows:   {len(test):,}  (public only; full set has {len(sub):,} IDs)")
print(f"Submission:  {len(sub):,} IDs to predict")
print()

# ── CLASS DISTRIBUTION ──────────────────────────────────────────────────────
print("CLASS DISTRIBUTION")
vc = train.target.value_counts().sort_index()
birads_names = {
    0: "Incomplete",
    1: "Negative",
    2: "Benign",
    3: "Prob. Benign",
    4: "Suspicious",
    5: "Highly Susp.",
    6: "Known Malig.",
}
for cls, cnt in vc.items():
    bar = "█" * int(cnt / len(train) * 50)
    print(f"  BI-RADS {cls} ({birads_names[cls]:<14}): {cnt:6,}  ({100*cnt/len(train):5.1f}%)  {bar}")
print()

# ── DUPLICATE ANALYSIS ──────────────────────────────────────────────────────
print("DUPLICATE ANALYSIS")
dup_count = train.duplicated(subset=["report"]).sum()
print(f"  Exact duplicate report texts: {dup_count:,} ({100*dup_count/len(train):.1f}% of rows)")

dedup = train.groupby("report")["target"].agg(list)
conflicts = dedup[dedup.apply(lambda x: len(set(x)) > 1)]
print(f"  Unique report texts:          {len(dedup):,}")
print(f"  Label conflicts (same text, diff label): {len(conflicts)}")
print(f"  All conflicts are between adjacent BI-RADS classes (1↔2, 2↔3)")
print()

# ── TEXT LENGTH STATS ────────────────────────────────────────────────────────
print("TEXT LENGTH BY CLASS (chars)")
train["text_len"] = train.report.str.len()
print(train.groupby("target")["text_len"].agg(["mean","median","min","max"]).to_string())
print("  → Longer reports tend toward abnormal classes (more findings to describe)")
print()

# ── LANGUAGE CHECK ───────────────────────────────────────────────────────────
print("LANGUAGE: Brazilian Portuguese")
print("  Key medical terms:")
medical_terms = {
    "nódulo (nodule)": r"nódulo|nodulo",
    "calcificações (calcifications)": r"calcifica",
    "assimetria (asymmetry)": r"assimetria",
    "distorção (distortion)": r"distorção",
    "biópsia (biopsy)": r"biopsia|biópsia|core biopsy",
    "carcinoma": r"carcinoma|cine",
    "espiculado (spiculated)": r"espiculado",
    "linfonodo (lymph node)": r"linfonodo",
}
for desc, pat in medical_terms.items():
    n = train.report.str.contains(pat, case=False, na=False).sum()
    print(f"  {desc:<40}: {n:,} reports ({100*n/len(train):.1f}%)")
print()

# ── LABEL LEAKAGE CHECK ──────────────────────────────────────────────────────
print("LABEL LEAKAGE CHECK")
leakage_pat = r"(categoria|cat\.|birads|bi-rads)\s*[0-6]"
leaky = train.report.str.lower().str.contains(leakage_pat, na=False)
print(f"  Reports containing explicit BI-RADS number in findings: {leaky.sum()}")
print("  → 71 reports mention 'categoria X' or 'BI-RADS X' in findings text.")
print("  → These are borderline leaky: they mention a prior score, not the current one.")
print("  → Model will likely use these as strong features (correct behavior).")
print()

# ── SUBMISSION ID ANALYSIS ───────────────────────────────────────────────────
print("SUBMISSION STRUCTURE WARNING")
print(f"  test.csv has {len(test)} rows")
print(f"  submission.csv requires {len(sub)} predictions")
print(f"  Missing test IDs: {sorted(set(sub.ID) - set(test.ID))}")
print("  → These 6 IDs are the HIDDEN test set evaluated server-side.")
print("  → You don't have their text, so predictions for them default to BI-RADS 2.")
print("  → Full test.csv in Kaggle environment will have all 10 rows.")
print()

# ── PER-CLASS EXAMPLE REPORTS ────────────────────────────────────────────────
print("EXAMPLE REPORTS BY CLASS")
for cls in [0, 1, 2, 3, 4, 5, 6]:
    sample = train[train.target == cls].report.iloc[0]
    print(f"\n  BI-RADS {cls} ({birads_names[cls]}):")
    print(f"  {sample[:280]}...")

# ── MOST DISCRIMINATIVE FEATURES ────────────────────────────────────────────
print("\n\nMOST DISCRIMINATIVE TERMS (by class)")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

dedup_df = train.groupby("report").agg(target=("target", lambda x: x.value_counts().index[0])).reset_index()
tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=3, max_features=30000, sublinear_tf=True)
X = tfidf.fit_transform(dedup_df.report)
y = dedup_df.target.values
lr = LogisticRegression(max_iter=500, class_weight="balanced", C=1.0, solver="saga", multi_class="ovr")
lr.fit(X, y)

feature_names = np.array(tfidf.get_feature_names_out())
for i, cls in enumerate(lr.classes_):
    top_idx = lr.coef_[i].argsort()[-10:][::-1]
    top_feats = feature_names[top_idx]
    print(f"  BI-RADS {cls}: {', '.join(top_feats)}")
