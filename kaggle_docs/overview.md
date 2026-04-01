# SPR 2026 Mammography Report Classification Overview

## Competition Goal

The **Radiology Society of São Paulo (SPR)** is hosting an AI challenge focused on breast cancer care.

The task is to build a model that predicts the **BI-RADS category** from the text of mammography reports.

A mammography report usually has three parts:

* **Indication** — why the exam was performed
* **Findings** — what the radiologist observed
* **Impression / Conclusion** — final BI-RADS category and recommendations

The dataset does **not** include the impression section because that contains the label to be predicted.

This is therefore a **text classification** problem where the model must infer the BI-RADS category from the report content alone.

---

## Why This Competition Matters

This challenge is meant to support:

* AI-driven mammography analysis
* more consistent radiology reporting
* educational tools for radiology trainees
* large-scale research in breast imaging and NLP

A strong solution could contribute to clinical decision-support systems that help radiologists and improve report quality.

---

## Problem Description

Mammography is one of the most effective tools for early breast cancer detection.

Radiologists use the **BI-RADS** system to standardize mammography reporting and communicate the degree of suspicion for malignancy and the recommended next step.

In this competition, participants must predict the BI-RADS class using only the **text of the radiology report**, excluding the impression/conclusion section.

Each report may include descriptive information such as:

* breast composition
* calcifications
* masses
* asymmetries
* associated radiologic features

The model must reason from the findings text rather than directly reading the final assessment.

---

## BI-RADS Categories

The target is one of seven classes:

| Category | Description                                     | Typical Management                 |
| -------- | ----------------------------------------------- | ---------------------------------- |
| 0        | Incomplete – Need additional imaging evaluation | Recall for additional imaging      |
| 1        | Negative                                        | Routine screening                  |
| 2        | Benign finding(s)                               | Routine screening                  |
| 3        | Probably benign                                 | Short-interval follow-up suggested |
| 4        | Suspicious abnormality                          | Consider biopsy                    |
| 5        | Highly suggestive of malignancy                 | Appropriate action should be taken |
| 6        | Known biopsy-proven malignancy                  | Surgical or oncologic treatment    |

Reference: American College of Radiology BI-RADS system

---

## Modeling Task

Participants receive:

* a **training set** of mammography reports with BI-RADS labels
* a **test set** of reports without labels

The goal is to output a predicted BI-RADS category for each test report.

This is a **7-class multiclass text classification** problem.

---

## Evaluation

Submissions are evaluated using:

**Macro F1-score**

This means performance matters across **all classes**, not just the most common ones.

That makes class imbalance important and means weak performance on rare classes can drag down the overall score.

---

## Submission Format

The required submission file format is:

```csv
id,target
1,6
2,4
3,1
4,2
```

* `id` = report identifier
* `target` = predicted BI-RADS class in `{0,1,2,3,4,5,6}`

Even though the overview text says “predict a probability,” the displayed submission format is a **single predicted class per report**.

---

## Prizes

Total prizes available: **$3,000**

* **1st place:** $1,500
* **2nd place:** $1,000
* **3rd place:** $500

---

## Code Submission Requirements

Submissions must be made through **Kaggle Notebooks**.

For the submit button to be active after commit:

* CPU notebook runtime must be **<= 9 hours**
* GPU notebook runtime must be **<= 9 hours**
* internet access must be **disabled**

This means the final solution must be reproducible inside Kaggle’s notebook environment.

---

## Timeline

* **Start:** about 2 months ago
* **Close:** about 1 month to go

---

## Additional Notes

The top three winners will be acknowledged during **Jornada Paulista de Radiologia (JPR) 2026**.

SPR is one of the major radiology organizations in Latin America and organizes the São Paulo Radiological Meeting, which adds some professional credibility to this competition.
