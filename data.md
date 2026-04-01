# SPR 2026 Mammography Report Classification Data

## Dataset Summary

This dataset contains **de-identified mammography radiology reports** collected from multiple Brazilian institutions.

The goal is to predict the final **BI-RADS category (0 to 6)** from the report text.

Unlike prior editions, this challenge does **not** include images. All predictive information comes from the **report text alone**.

---

## Privacy and Data Restrictions

The dataset contains **de-identified patient data**.

As a participant, you must:

* not redistribute the data
* not attempt re-identification
* not probe the test set labels
* report any accidental personally identifiable information to the organizers

For repo notes, the important practical takeaway is:

* treat this as restricted competition data
* do not upload raw report text publicly outside allowed competition rules

---

## Prediction Target

This is a **multiclass classification** problem with **7 classes**:

| Class | Meaning                                             |
| ----- | --------------------------------------------------- |
| 0     | Incomplete, needs additional imaging                |
| 1     | Negative                                            |
| 2     | Benign finding                                      |
| 3     | Probably benign, short-interval follow-up suggested |
| 4     | Suspicious abnormality                              |
| 5     | Highly suggestive of malignancy                     |
| 6     | Known biopsy-proven malignancy                      |

The label was extracted from the original **impression / conclusion** section of the report.

---

## Report Structure

Reports typically follow three sections:

1. **Indication / Reason for Exam**
   Why the exam was performed, such as screening, clinical complaint, or follow-up.

2. **Findings**
   Radiologist description of mammographic findings, often using BI-RADS vocabulary and laterality.

3. **Impression / Conclusion**
   Final BI-RADS category and possible recommendation.

For this competition, only the **indication** and **findings** are provided as model input.

The **impression** is removed because it contains the exact target.

---

## Files

## `train.csv`

Contains labeled training examples.

### Columns

* `id` — unique report identifier
* `report` — report text including indication and findings, without the impression
* `target` — BI-RADS class in `{0,1,2,3,4,5,6}`

### Example

```csv
id,report,target
000001,"INDICATION: Screening. FINDINGS: ...",2
```

## `test.csv`

Contains the reports for which the BI-RADS category must be predicted.

### Columns

* `id`
* `report` — indication and findings only

No labels are provided.

## `submission.csv`

Example submission file.

### Columns

* `id`
* `target` — predicted BI-RADS class in `{0,1,2,3,4,5,6}`

### Example

```csv
id,target
Acc1,2
Acc2,0
Acc3,4
```

---

## Core Modeling Implications

This is a **medical NLP classification task** with several important implications:

* input is **free-text radiology report language**
* target is **7-class BI-RADS**
* the impression section is removed, so the model must infer the label from findings
* class imbalance may matter because macro F1 is the metric
* medical terms, laterality, lesion descriptions, and diagnostic wording are likely important

---

## Competition Rules That Matter for Modeling

Important practical rules:

* **No manual labeling of the test set**
* **Do not attempt re-identification**
* the **final solution must be open-sourced** in the discussion forum
* external data or models must follow competition rules and be declared

That means your final pipeline should be:

* reproducible
* clean enough to publish
* based on allowed resources only

---

## File Summary

* **Number of files:** 3
* `train.csv`
* `test.csv`
* `submission.csv`

Dataset size shown on Kaggle: **7.94 MB**

License: **Subject to Competition Rules**

---

## Practical Takeaway

From a modeling perspective, this dataset is basically:

* **input:** report text
* **output:** BI-RADS class
* **problem type:** multiclass text classification
* **metric pressure:** good performance across all classes, not just the most common ones
* **best likely baseline family:** TF-IDF + linear models, then transformer-based fine-tuning if worthwhile
