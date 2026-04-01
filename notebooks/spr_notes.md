# SPR 2026 Competition Notes

## Quick CV Results
| Model | Macro F1 (dedup-aware CV) | Notes |
|---|---|---|
| TF-IDF word 1-3gram + LR balanced | 0.49 | LR is weak here |
| TF-IDF word 1-3gram + LinearSVC balanced C=0.3 | 0.70 | Best sparse |
| Combined word+char ngrams + LinearSVC | **0.711** | Current best |
| Portuguese BERT / XLM-RoBERTa (estimated) | ~0.80+ | Not yet run |

## Key Data Facts
- 18,272 training rows; 9,131 unique report texts (50% exact duplicates)
- 87.4% BI-RADS 2 (benign) — extreme imbalance
- BI-RADS 5 (highly suspicious): only 29 examples; hardest class
- Text is Brazilian Portuguese
- Reports are templated — identical boilerplate text for normal screenings
- 11 report texts have conflicting labels (all BI-RADS 1 vs 2 borderline)
- Test set: 10 IDs in submission.csv; only 4 visible in test.csv

## Hardest Confusions (SVC OOF)
- BI-RADS 3 ↔ BI-RADS 0: common confusion (both involve ambiguous/incomplete findings)
- BI-RADS 5 ↔ BI-RADS 4/6: very hard with only 29 examples
- BI-RADS 3 ↔ BI-RADS 2: benign vs probably benign boundary
