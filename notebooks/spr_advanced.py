"""
SPR 2026 Mammography Report Classification
Advanced Pipeline: Portuguese BERT / XLM-RoBERTa fine-tuning

Requirements on Kaggle:
- Add dataset: neuralmind/bert-base-portuguese-cased  (or xlm-roberta-base)
- GPU runtime (T4 or P100)
- Disable internet access during submission

Expected CV improvement: ~0.71 (sparse) → ~0.80-0.85 (transformer)
The gains come from:
  1. Subword tokenization handles Portuguese morphology better than TF-IDF
  2. Pre-trained medical language patterns in multilingual models
  3. Contextual representations vs bag-of-ngrams
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Option A (best for this task): Portuguese BERT
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # needs offline dataset

# Option B (easier Kaggle setup): Multilingual
# MODEL_NAME = "xlm-roberta-base"

# Option C (fastest, decent): multilingual MiniLM
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MAX_LEN    = 256   # 95th percentile of tokens; BI-RADS text is short
BATCH_SIZE = 16
EPOCHS     = 4
LR         = 2e-5
N_FOLDS    = 5
N_CLASSES  = 7
SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_WEIGHTS = torch.tensor(
    [4.279, 3.767, 0.163, 3.661, 12.198, 90.010, 58.006],
    dtype=torch.float
).to(DEVICE)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

# ─── DATA ────────────────────────────────────────────────────────────────────
train = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/train.csv")
test  = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/test.csv")
sub   = pd.read_csv("/kaggle/input/spr-2026-mammography-report-classification/submission.csv")

# Dedup for CV (same reasoning as baseline)
dedup = (
    train
    .groupby("report")
    .agg(target=("target", lambda x: x.value_counts().index[0]))
    .reset_index()
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ─── DATASET ─────────────────────────────────────────────────────────────────
class MammoDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=MAX_LEN):
        self.texts  = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbs  = batch["labels"].to(DEVICE)
        out  = model(input_ids=ids, attention_mask=mask)
        loss = criterion(out.logits, lbs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict(model, loader):
    model.eval()
    preds, probas = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            out  = model(input_ids=ids, attention_mask=mask)
            proba = torch.softmax(out.logits, dim=-1).cpu().numpy()
            probas.append(proba)
            preds.extend(proba.argmax(axis=1))
    return np.array(preds), np.vstack(probas)


# ─── K-FOLD CROSS-VALIDATION ─────────────────────────────────────────────────
X_cv = dedup["report"].values
y_cv = dedup["target"].values
X_full = train["report"].fillna("").values
y_full = train["target"].values
X_test = test["report"].fillna("").values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds  = np.zeros(len(X_cv), dtype=int)
oof_probas = np.zeros((len(X_cv), N_CLASSES))
test_probas_all = np.zeros((len(X_test), N_CLASSES))

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

    tr_ds  = MammoDataset(X_cv[tr_idx], y_cv[tr_idx])
    val_ds = MammoDataset(X_cv[val_idx], y_cv[val_idx])
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=0.01
    )
    total_steps = len(tr_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_probas = None
    for epoch in range(EPOCHS):
        loss = train_epoch(model, tr_dl, optimizer, scheduler, criterion)
        val_preds, val_probas = predict(model, val_dl)
        f1 = f1_score(y_cv[val_idx], val_preds, average="macro")
        print(f"  Epoch {epoch+1}: loss={loss:.4f}  val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_probas = val_probas.copy()

    oof_preds[val_idx]  = best_probas.argmax(axis=1)
    oof_probas[val_idx] = best_probas

    # Test inference on this fold
    test_ds = MammoDataset(X_test)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False)
    _, test_fold_probas = predict(model, test_dl)
    test_probas_all += test_fold_probas / N_FOLDS

    del model
    torch.cuda.empty_cache()

# ─── FINAL METRICS ───────────────────────────────────────────────────────────
cv_f1 = f1_score(y_cv, oof_preds, average="macro")
print(f"\n=== CV Macro F1 (dedup-aware): {cv_f1:.4f} ===")
print(classification_report(y_cv, oof_preds, digits=3))

# ─── SUBMISSION ──────────────────────────────────────────────────────────────
test_labels = test_probas_all.argmax(axis=1)
pred_map    = dict(zip(test["ID"].values, test_labels))
sub["target"] = sub["ID"].map(pred_map).fillna(2).astype(int)
sub.to_csv("submission.csv", index=False)
print("\nSaved: submission.csv")
print(sub.to_string(index=False))
