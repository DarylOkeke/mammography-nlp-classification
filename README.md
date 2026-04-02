# Mammography Report Classification

Automated BI-RADS category prediction from free-text radiology reports using NLP.

## Overview

Radiologists classify mammography findings using the **BI-RADS** (Breast Imaging Reporting and Data System) scale — a standardized 7-category system ranging from incomplete/recall (0) to biopsy-proven malignancy (6). Assigning the correct BI-RADS category from a written report is a high-stakes task: misclassification can delay cancer diagnosis or trigger unnecessary procedures.

This project builds an NLP pipeline that reads the **indication** and **findings** sections of a mammography report — without access to the radiologist's final impression — and predicts the BI-RADS category. The dataset consists of de-identified Brazilian Portuguese reports collected across multiple clinical institutions.

## BI-RADS Categories

| Category | Meaning | Clinical Action |
|----------|---------|----------------|
| 0 | Incomplete — additional imaging needed | Recall patient |
| 1 | Negative | Routine screening |
| 2 | Benign finding | Routine screening |
| 3 | Probably benign | Short-interval follow-up |
| 4 | Suspicious abnormality | Consider biopsy |
| 5 | Highly suggestive of malignancy | Biopsy indicated |
| 6 | Known biopsy-proven malignancy | Surgical/oncologic treatment |

## Problem Formulation

- **Task:** 7-class multiclass text classification
- **Input:** Portuguese radiology report text (indication + findings sections)
- **Output:** BI-RADS category `{0, 1, 2, 3, 4, 5, 6}`
- **Metric:** Macro F1-Score (equal weight across all classes — rare classes matter as much as common ones)
- **Language:** Brazilian Portuguese medical text
- **Training set:** 18,272 reports

**The key challenge:** The dataset is severely imbalanced — 87.4% of reports are BI-RADS 2 (benign). Classes 5 and 6 have 29 and 45 examples respectively. The macro F1 metric forces the model to perform well across all classes, including the rarest ones.

## Dataset

De-identified patient data collected from multiple Brazilian radiology institutions. Reports follow a structured format:

1. **Indicação clínica** — reason for the exam (screening, follow-up, clinical complaint)
2. **Achados** — radiologist's findings (composition, calcifications, nodules, asymmetries)
3. **Análise comparativa** — comparison to prior exams

The impression/conclusion section (which directly states the BI-RADS category) is withheld. The model must infer the category from findings alone.

## Approach

### Baseline: TF-IDF + Logistic Regression

**Feature extraction:**
- Word n-grams (1–2): captures unigrams and compound clinical phrases ("nódulo espiculado", "calcificações pleomórficas")
- Character n-grams (3–5): handles Portuguese morphological variation ("espiculado/espiculada/espiculados")
- Stacked via `FeatureUnion` to give the classifier both signals simultaneously

**Classifier:**
- Logistic Regression with `class_weight='balanced'`
- Automatically upweights rare classes proportional to their inverse frequency
- Class 5 (29 examples) receives ~43× the weight of class 2 (15,968 examples)

**Validation:**
- 5-fold stratified cross-validation
- Stratification ensures each fold has proportional representation of all 7 classes, including class 5 with only 29 examples

### Baseline Results

| Class | F1 Score | Support |
|-------|----------|---------|
| BI-RADS 0 | 0.55 | 610 |
| BI-RADS 1 | 0.85 | 693 |
| BI-RADS 2 | 0.93 | 15,968 |
| BI-RADS 3 | 0.36 | 713 |
| BI-RADS 4 | 0.26 | 214 |
| BI-RADS 5 | 0.09 | 29 |
| BI-RADS 6 | 0.17 | 45 |
| **Macro F1** | **0.46** | 18,272 |

The model learns clinically meaningful vocabulary — top predictors for each class align with established BI-RADS descriptors:
- Class 4: "calcificações pleomórficas" (pleomorphic calcifications)
- Class 5: "retração", "espiculada" (retraction, spiculated morphology)
- Class 6: "carcinoma", "resultado de biópsia" (biopsy result)

### Planned Improvements

1. **Portuguese BERT fine-tuning** — `neuralmind/bert-base-portuguese-cased` (BERTimbau) for contextual language understanding
2. **Manual class weight tuning** — optimize weights for classes 5 and 6 beyond the automatic `balanced` setting
3. **Section-aware features** — parse indication and findings as separate feature streams; the indication (screening vs. cancer follow-up) is independently discriminative

## Repo Structure

```
competition_data/         Raw data (train.csv, test.csv)
notebooks/
  baseline.ipynb          TF-IDF + LogReg baseline pipeline
kaggle_docs/              Competition documentation
```

## Key Technical Decisions

**Why minimal text preprocessing?**
Portuguese clinical negation is load-bearing. "Não se observam nódulos" (no nodules observed) is the defining phrase of a normal report — stripping stopwords or the word "não" destroys this signal. TF-IDF's IDF weighting handles uninformative high-frequency terms without explicit removal.

**Why char n-grams alongside word n-grams?**
Portuguese inflects words by gender and number ("espiculado/espiculada/espiculados"). A word-level model treats these as three separate features. Character n-grams share the "espicul" substring, letting the model generalize across morphological variants.

**Why balanced class weights?**
Without them, logistic regression's loss function is dominated by class 2 (87.4% of data). The model learns to predict class 2 for most inputs and scores near-zero F1 on classes 4–6. Balanced weighting forces the model to treat a mistake on class 5 as ~43× more costly than a mistake on class 2.

## Data Notes

- Reports are in Brazilian Portuguese and follow structured clinical formatting
- Dates are de-identified with the placeholder `<DATA>` — its presence signals a follow-up exam
- ~50% of training rows are exact text duplicates (boilerplate normal screening reports)
- 11 text groups have conflicting labels across different radiologists — clinically expected for borderline BI-RADS 1 vs. 2 cases

## License

Data is subject to competition rules and patient privacy restrictions. Code is MIT licensed.
