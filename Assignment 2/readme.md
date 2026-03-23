# Assignment 2: Text Classification
**Student:** Josh  
**Course:** BSAN 6080 — Text Mining  
**Date:** March 2026  
**Dataset:** Fake and Real News (Kaggle)

---

## Overview
Binary text classification task to distinguish fake news articles from real news 
articles. Each article receives exactly one label — Fake (0) or Real (1). Models 
are trained from scratch using supervised machine learning with TF-IDF features, 
unlike Assignment 1 which used pretrained models.

---

## Dataset Details
- **Source:** https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- **Size:** 44,898 articles (21,417 real + 23,481 fake)
- **Classes:** 2 — Fake (0) and Real (1)
- **Distribution:** 47.7% Real / 52.3% Fake — approximately balanced
- **Features used:** Article text body (cleaned and lemmatized)

---

## Preprocessing Approach
- Lowercased all text
- Removed URLs, emails, numbers, and punctuation
- Filtered English stopwords
- Lemmatized remaining tokens
- Removed tokens shorter than 3 characters
- Feature representation: TF-IDF with 5,000 features, unigrams and bigrams

---

## Best Model Results
| Metric | Fake Class | Real Class |
|---|---|---|
| Precision | 0.9979 | 0.9970 |
| Recall | 0.9972 | 0.9977 |
| F1 | 0.9976 | 0.9973 |
| **Weighted F1** | **0.9974** | |

**Model:** Logistic Regression (Tuned) + TF-IDF  
**Tuning:** GridSearchCV, 5-fold CV, f1_weighted scoring

---

## Important Class
**Both classes equally important — optimize for balance (Weighted F1)**

Fake news detection carries significant costs in both directions. Missed 
detections allow misinformation to reach readers. False alarms suppress 
legitimate journalism. Unlike spam detection where missed detections dominate, 
fake news requires balance across both error types. Weighted F1 captures this 
by penalizing models that are strong on one class but weak on the other.

---

## Model Comparison (5 Criteria)
| Criterion | Logistic Regression (Tuned) | Linear SVM | Naive Bayes |
|---|---|---|---|
| **Weighted F1** | 0.9974 (best) | 0.9973 | 0.9638 |
| **Speed** | Moderate | Fast | Fastest |
| **Fake Class F1** | 0.9976 (best) | 0.9975 | 0.9650 |
| **Interpretability** | High — feature coefficients | Moderate | High |
| **Ease of Tuning** | Easy — GridSearchCV on C | Moderate | Minimal |

---

## Custom Inference Summary
- **13 out of 20 correct (65%)** on fresh AI-generated examples
- Perfect on obvious fake examples — 5 out of 5
- Struggled on tricky real articles — 2 out of 5
- Partial on out-of-domain examples — 3 out of 5
- Key finding: model relies on source writing style patterns rather than 
  genuine misinformation signals, explaining the gap between test F1 and 
  inference accuracy

---

## Recommendation
**Logistic Regression (Tuned) + TF-IDF** is the recommended model. It achieved 
the highest weighted F1, offers direct interpretability through feature 
coefficients, and performed strongest on both classes after tuning. Not 
recommended for production without retraining on more stylistically diverse 
data — current performance likely reflects source-level leakage rather than 
robust fake news detection.
