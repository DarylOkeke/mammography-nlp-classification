# =============================================================================
# SPR 2026 Mammography Report Classification — Baseline
# =============================================================================
#
# PROBLEM: Predict the BI-RADS category (0–6) from Portuguese mammography
#          report text. The "impression" section (which contains the answer)
#          is withheld; we must infer the label from the indication + findings.
#
# METRIC:  Macro F1-Score — the F1 is computed per class and then averaged
#          equally across all 7 classes. This means rare classes (BI-RADS 4,5,6)
#          matter just as much as the dominant class 2. A model that only
#          predicts class 2 correctly would score near 0.
#
# APPROACH (this file):
#   1. TF-IDF vectorization (word n-grams + char n-grams, stacked together)
#   2. Logistic Regression with balanced class weights
#   3. Stratified 5-Fold cross-validation → honest macro F1 estimate
#   4. Per-class analysis to see where the model struggles
#   5. Generate a valid submission file
#
# WHY TF-IDF + LR AS BASELINE:
#   - Fast, interpretable, and often surprisingly competitive for short structured text
#   - TF-IDF captures the n-gram patterns that strongly discriminate BI-RADS classes
#   - Logistic Regression is well-calibrated and handles class imbalance cleanly
#   - Once we have an honest CV score, we know the bar that transformers must beat
#
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1: PATH CONFIGURATION
# =============================================================================
# This block auto-detects whether we're running locally or on Kaggle.
# On Kaggle, the data lives at /kaggle/input/<competition-slug>/
# Locally, we use the path relative to this script.

KAGGLE_INPUT = "/kaggle/input/spr-2026-mammography-report-classification"
LOCAL_INPUT  = os.path.join(os.path.dirname(__file__), "..", "competition_data")

if os.path.exists(KAGGLE_INPUT):
    DATA_DIR = KAGGLE_INPUT
    print("Running on Kaggle.")
else:
    DATA_DIR = LOCAL_INPUT
    print("Running locally.")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")
SUB_PATH   = os.path.join(DATA_DIR, "submission.csv")

# Number of cross-validation folds.
# 5 is standard. With 29 examples in the rarest class, 5 folds = ~5 per fold.
N_FOLDS = 5

# Random seed for reproducibility.
SEED = 42


# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================

def load_data():
    """Load train and test CSVs. Return dataframes with verified column names."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    # Standardize column names to lowercase for convenience.
    train.columns = train.columns.str.strip().str.lower()
    test.columns  = test.columns.str.strip().str.lower()

    # Sanity checks — crash early rather than silently produce wrong results.
    assert "report" in train.columns, "Expected 'report' column in train"
    assert "target" in train.columns, "Expected 'target' column in train"
    assert "report" in test.columns,  "Expected 'report' column in test"

    print(f"Train shape: {train.shape}  |  Test shape: {test.shape}")
    print(f"Target classes: {sorted(train['target'].unique())}")
    return train, test


# =============================================================================
# SECTION 3: PREPROCESSING
# =============================================================================
# We do minimal text preprocessing here because:
#   1. TF-IDF handles tokenization internally.
#   2. Removing stopwords in Portuguese would destroy key clinical negations
#      like "não se observam" (not observed) — the word "não" (not) is
#      diagnostic: "Não se observam nódulos" (no nodules) → benign.
#   3. Special Portuguese characters (ã, é, ç) carry meaning — keep them.
#   4. The `<DATA>` placeholder for de-identified dates should be kept as-is
#      (it's a feature — its presence indicates a follow-up report).

def preprocess_text(text: str) -> str:
    """
    Light cleaning: lowercase only.
    We do NOT strip accents or remove punctuation because:
    - accents distinguish Portuguese words ("mama" vs common English)
    - "não" (not) is a critical clinical negation word — removing it would hurt
    """
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text preprocessing to the 'report' column in place."""
    df = df.copy()
    df["report"] = df["report"].apply(preprocess_text)
    return df


# =============================================================================
# SECTION 4: FEATURE EXTRACTION (TF-IDF)
# =============================================================================
# We stack two TF-IDF vectorizers:
#
# 1. WORD n-grams (1,2):
#    - Unigrams capture single keywords: "nódulo", "biopsia", "rastreamento"
#    - Bigrams capture compound clinical terms: "nódulo espiculado",
#      "calcificações pleomórficas", "distorção arquitetural"
#    - These bigrams are among the strongest BI-RADS class predictors.
#
# 2. CHAR n-grams (3,5):
#    - Character-level substrings handle morphological variants in Portuguese:
#      "espiculado" vs "espiculados", "calcificação" vs "calcificações"
#    - Also helps with minor OCR artifacts or non-standard spacing
#    - Less sensitive to word boundary issues
#
# WHY STACK THEM?
#    Word TF-IDF and char TF-IDF capture different aspects of the text.
#    Stacking them (via FeatureUnion) gives the classifier both signals at once.
#    This is a classic and robust approach for short medical text.

def build_tfidf_pipeline(classifier):
    """
    Build a full sklearn Pipeline that:
      1. Extracts stacked TF-IDF features (word + char n-grams)
      2. Passes them to the provided classifier

    Parameters
    ----------
    classifier : sklearn estimator
        Any sklearn classifier (LogisticRegression, LinearSVC, etc.)

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    # Word-level TF-IDF
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),     # unigrams + bigrams
        min_df=3,               # ignore terms appearing in fewer than 3 docs
                                # (avoids noise from very rare terms)
        max_df=0.95,            # ignore terms in >95% of docs (near-universal)
        sublinear_tf=True,      # apply log(1 + tf) to dampen very frequent terms
        strip_accents=None,     # keep Portuguese accents — they matter
        token_pattern=r"(?u)\b\w+\b",  # standard word tokenizer
    )

    # Character-level TF-IDF
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",     # char n-grams within word boundaries
        ngram_range=(3, 5),     # 3-, 4-, 5-character substrings
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )

    # Combine both vectorizers into a single feature matrix
    features = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf),
    ])

    pipeline = Pipeline([
        ("features", features),
        ("clf",      classifier),
    ])

    return pipeline


# =============================================================================
# SECTION 5: CROSS-VALIDATION
# =============================================================================
# We use Stratified K-Fold because:
#   - Class 5 has only 29 examples and class 6 has only 45.
#   - Without stratification, some folds might contain 0 examples of a rare class.
#   - Stratified folds ensure each fold has a proportional representation of each class.
#   - With 5 folds: class 5 gets ~5-6 examples per fold, class 6 gets ~9 per fold.
#
# IMPORTANT NOTE ON DUPLICATES:
#   The training set has ~9,928 exact duplicate reports (same text, same or different label).
#   These are real boilerplate mammography reports (e.g., all-normal screening exams).
#   In cross-validation, the same text could appear in both train and validation folds.
#   This might slightly inflate CV scores. A more conservative approach would be to
#   deduplicate before CV splitting, but for baseline purposes we keep them.
#   See Part 6 for this as a flagged ambiguous decision.

def run_cross_validation(pipeline, X: pd.Series, y: pd.Series, n_folds: int = 5):
    """
    Run stratified cross-validation and return per-fold scores and OOF predictions.

    OOF = Out-Of-Fold predictions: each sample gets a prediction from a model
    that never saw it during training. This is the gold-standard way to estimate
    how your model will perform on unseen data.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X        : pd.Series — report text
    y        : pd.Series — true BI-RADS labels
    n_folds  : int

    Returns
    -------
    oof_preds : np.ndarray — predicted labels for every training example
    fold_scores : list of float — macro F1 per fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    oof_preds   = np.zeros(len(y), dtype=int)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        oof_preds[val_idx] = preds

        fold_f1 = f1_score(y_val, preds, average="macro")
        fold_scores.append(fold_f1)
        print(f"  Fold {fold_idx + 1}/{n_folds}  —  macro F1: {fold_f1:.4f}")

    return oof_preds, fold_scores


# =============================================================================
# SECTION 6: EVALUATION AND ANALYSIS
# =============================================================================

def evaluate(y_true, y_pred, label=""):
    """
    Print a full classification report and macro F1.
    The classification report shows precision, recall, and F1 per class.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS {label}")
    print(f"{'=' * 60}")
    print(f"Overall Macro F1: {macro_f1:.4f}")
    print()

    # Per-class breakdown
    # Precision = of all predictions for class X, how many were correct?
    # Recall    = of all true class X examples, how many did we find?
    # F1        = harmonic mean of precision and recall (balances both)
    # Support   = number of true examples of that class
    print(classification_report(
        y_true, y_pred,
        labels=sorted(set(y_true) | set(y_pred)),
        target_names=[f"BI-RADS {c}" for c in sorted(set(y_true) | set(y_pred))],
        zero_division=0,
    ))
    return macro_f1


def print_confusion_matrix(y_true, y_pred):
    """
    Print a confusion matrix.
    Row = true class, Column = predicted class.
    Diagonal = correct predictions.
    Off-diagonal = errors and which class we confused it with.
    """
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("\nCONFUSION MATRIX (rows = true, cols = predicted)")
    print(f"Classes: {classes}")

    # Header
    header = "True\\Pred |" + "".join(f" BR{c:1d} " for c in classes)
    print(header)
    print("-" * len(header))
    for i, c in enumerate(classes):
        row_str = f" BR{c:1d}  True |"
        row_str += "".join(f" {cm[i, j]:4d} " for j in range(len(classes)))
        print(row_str)

    print()
    # Highlight the most common error types
    print("COMMON ERRORS (top misclassifications):")
    for i, true_c in enumerate(classes):
        for j, pred_c in enumerate(classes):
            if i != j and cm[i, j] > 0:
                pct = cm[i, j] / cm[i].sum() * 100
                print(f"  True BI-RADS {true_c} → predicted BI-RADS {pred_c}: "
                      f"{cm[i, j]:4d} cases ({pct:.1f}% of class {true_c})")


def analyze_errors(train_df, oof_preds):
    """
    Print examples of misclassified reports to build intuition.
    Shows the first few words of incorrectly predicted reports.
    """
    wrong_mask = train_df["target"].values != oof_preds
    errors = train_df[wrong_mask].copy()
    errors["predicted"] = oof_preds[wrong_mask]
    errors["preview"] = errors["report"].str[:150]

    print(f"\nMISCLASSIFIED: {wrong_mask.sum()} / {len(train_df)} "
          f"({wrong_mask.mean() * 100:.1f}%)")
    print()

    # Show one example per true class where we got it wrong
    for true_class in sorted(errors["target"].unique()):
        subset = errors[errors["target"] == true_class].head(1)
        for _, row in subset.iterrows():
            print(f"TRUE: BI-RADS {row['target']}  →  PREDICTED: BI-RADS {row['predicted']}")
            print(f"  TEXT: {row['preview']}...")
            print()


# =============================================================================
# SECTION 7: SUBMISSION GENERATION
# =============================================================================

def generate_submission(pipeline, test_df, output_path="submission_baseline.csv"):
    """
    Predict on the test set and write a valid submission file.

    The submission format requires:
        ID, target
        Acc0, 2
        Acc2, 1
        ...

    NOTE: The local test.csv only has 4 rows. On Kaggle, it will have more.
    This function handles whatever rows are present in test.csv.
    """
    test_preds = pipeline.predict(test_df["report"])

    submission = pd.DataFrame({
        "ID":     test_df["id"],   # column was lowercased during load
        "target": test_preds,
    })

    # Write next to this script if running locally, otherwise to /kaggle/working/
    if os.path.exists("/kaggle/working"):
        out_path = f"/kaggle/working/{output_path}"
    else:
        out_path = os.path.join(os.path.dirname(__file__), output_path)

    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path}")
    print(submission.head(10))
    return submission


# =============================================================================
# SECTION 8: MAIN — PUT IT ALL TOGETHER
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("SPR 2026 Mammography Report Classification — Baseline")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------
    # 8.1 Load data
    # -------------------------------------------------------------------
    train_df, test_df = load_data()

    # -------------------------------------------------------------------
    # 8.2 Quick EDA summary (facts we confirmed during data audit)
    # -------------------------------------------------------------------
    print("\n--- Data Summary ---")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples:     {len(test_df)}")
    print(f"\nClass distribution:")
    total = len(train_df)
    for cls, cnt in train_df["target"].value_counts().sort_index().items():
        bar = "#" * int(cnt / total * 50)
        print(f"  BI-RADS {cls}: {cnt:5d} ({cnt/total*100:5.1f}%)  {bar}")

    # Check for duplicate reports — affects CV honesty
    n_dup_texts = train_df["report"].duplicated().sum()
    print(f"\nDuplicate report texts: {n_dup_texts} ({n_dup_texts/total*100:.1f}%)")
    print("(These are boilerplate normal reports; kept in training data for now.)")

    # -------------------------------------------------------------------
    # 8.3 Preprocess
    # -------------------------------------------------------------------
    print("\n--- Preprocessing ---")
    train_df = preprocess_dataframe(train_df)
    test_df  = preprocess_dataframe(test_df)
    print("Applied: lowercase. No stopword removal (clinical negations matter).")

    X_train = train_df["report"]
    y_train = train_df["target"]

    # -------------------------------------------------------------------
    # 8.4 Build the model
    # -------------------------------------------------------------------
    # Logistic Regression with balanced class weights.
    #
    # class_weight='balanced':
    #   sklearn automatically assigns each class a weight inversely proportional
    #   to its frequency. So class 5 (29 examples) gets ~635x the weight of
    #   class 2 (15,968 examples). This prevents the model from ignoring rare
    #   classes — critical for macro F1.
    #
    # C=1.0 (default regularization strength):
    #   Controls how strongly we penalize large coefficients.
    #   Smaller C = more regularization = simpler model.
    #   C=1.0 is a reasonable starting default. Tune later if needed.
    #
    # max_iter=1000:
    #   Logistic regression uses an iterative solver. Default 100 often doesn't
    #   converge for large feature spaces. 1000 is safe.
    #
    # solver='saga':
    #   Best solver for large sparse feature matrices (which TF-IDF produces).
    #   Handles multinomial multi-class well.

    classifier = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="saga",
        multi_class="multinomial",
        random_state=SEED,
        n_jobs=-1,
    )

    pipeline = build_tfidf_pipeline(classifier)

    # -------------------------------------------------------------------
    # 8.5 Cross-validation
    # -------------------------------------------------------------------
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    print("(Each fold trains on 80% and validates on 20%, stratified by class.)\n")

    oof_preds, fold_scores = run_cross_validation(pipeline, X_train, y_train, N_FOLDS)

    print(f"\nFold macro F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"Mean macro F1: {np.mean(fold_scores):.4f}  ±  {np.std(fold_scores):.4f}")

    # -------------------------------------------------------------------
    # 8.6 Full OOF evaluation (across all folds combined)
    # -------------------------------------------------------------------
    # OOF = Out-of-Fold: each prediction was made by a model that never saw that row.
    # This gives us the most honest estimate of real-world performance.
    macro_f1 = evaluate(y_train, oof_preds, label="(OOF — all folds combined)")
    print_confusion_matrix(y_train, oof_preds)
    analyze_errors(train_df, oof_preds)

    # -------------------------------------------------------------------
    # 8.7 Retrain on full training set and predict test
    # -------------------------------------------------------------------
    # After cross-validation tells us the model is reasonable,
    # we retrain on ALL training data before predicting the test set.
    # More training data = better final model.
    print("\n--- Retraining on full training set ---")
    pipeline.fit(X_train, y_train)
    print("Done. Predicting test set...")

    submission = generate_submission(pipeline, test_df)

    # -------------------------------------------------------------------
    # 8.8 Print model insights: top features per class
    # -------------------------------------------------------------------
    print_top_features_per_class(pipeline, y_train)

    print("\n" + "=" * 60)
    print(f"FINAL BASELINE RESULT: Macro F1 = {macro_f1:.4f} (OOF)")
    print("=" * 60)
    return macro_f1, oof_preds


# =============================================================================
# SECTION 9: MODEL INTERPRETATION
# =============================================================================

def print_top_features_per_class(pipeline, y_train, top_n=10):
    """
    For Logistic Regression, we can inspect the learned coefficients to see
    which words/n-grams the model associates with each BI-RADS class.

    A large positive coefficient = strong signal for that class.
    This is one of the advantages of TF-IDF + LR over black-box models:
    we can understand WHAT the model learned.
    """
    print("\n--- TOP FEATURES PER CLASS ---")
    print("(Words/n-grams with largest positive coefficients per BI-RADS class)")
    print()

    clf = pipeline.named_steps["clf"]
    features = pipeline.named_steps["features"]

    # Reconstruct the full feature name list from the FeatureUnion
    word_names = features.transformer_list[0][1].get_feature_names_out()
    char_names = features.transformer_list[1][1].get_feature_names_out()
    all_feature_names = np.concatenate([word_names, char_names])

    # clf.coef_ has shape (n_classes, n_features) for multinomial LR
    classes = clf.classes_
    for i, cls in enumerate(classes):
        coef = clf.coef_[i]
        top_idx = np.argsort(coef)[-top_n:][::-1]
        top_terms = [(all_feature_names[j], coef[j]) for j in top_idx]

        print(f"  BI-RADS {cls} — top {top_n} features:")
        for term, weight in top_terms:
            print(f"    {term!r:40s}  coef={weight:.3f}")
        print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
