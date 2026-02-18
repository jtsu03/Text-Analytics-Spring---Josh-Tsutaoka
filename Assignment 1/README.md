# Sentiment Analysis Model Comparison
### Human Labels vs VADER vs TextBlob vs Transformer

---

## Project Overview
This project evaluates how well different sentiment analysis approaches interpret human opinion in text.  
Rather than relying on model outputs alone, I manually labeled sentiment to create a ground truth dataset and compared multiple NLP approaches against human judgment.

The objective is to determine whether more advanced models truly understand meaning better or simply produce different statistical predictions.

Models evaluated:
- VADER (rule-based sentiment)
- TextBlob (polarity scoring)
- HuggingFace Transformer (contextual deep learning)

---

## Dataset Description

### Source
The data comes from the Kaggle dataset:

**KUC Hackathon Winter 2018 – Drug Review Dataset**  
https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

---

### Original Dataset
The original dataset contains user written drug reviews collected from an online medical review platform.

Each record includes:

- `drugName` — medication name
- `condition` — condition being treated
- `review` / `text` — written user experience
- `rating` — numerical score from 1–10
- `date`
- `usefulCount` — number of helpful votes

The full dataset contains thousands of reviews with mixed sentiment strength.

---

### Dataset Used in This Project
For clearer sentiment evaluation, the dataset was filtered to include only extreme opinions:

- Rating **10 → Positive sentiment**
- Rating **1 → Negative sentiment**

This reduces ambiguity and allows models to be tested on clear emotional language before comparing them against human interpretation.

Then a subset of **100 reviews** was manually labeled to create a ground truth evaluation dataset.

---

### Final Evaluation Fields
Each record in the final dataset includes:

- `text` — original review text
- `human_label` — manually assigned sentiment
- `vader_label`
- `tb_label`
- `hf_label`
- model confidence scores

Human labels were created first and treated as the ground truth so model performance reflects agreement with human understanding rather than agreement with other models.

---

## Quantitative Results

| Model | Accuracy |
|------|------|
| **VADER** | **58%** |
| **TextBlob** | 55% |
| **Transformer** | 52% |

---

## Model Comparison

| Criterion | VADER | TextBlob | Transformer | Winner | Justification |
|------|------|------|------|------|------|
| Speed | 0.07 sec / 500 reviews | 0.05 sec / 500 reviews | 0.61 sec / 500 reviews | TextBlob | Lower runtime is faster |
| Accuracy (human labels) | 58% | 55% | 52% | VADER | Highest agreement with human labels |
| Handles emphasis (caps/punctuation) | Excellent | Weak | Good | VADER | Rule based intensity handling |
| Handles negation | Good | Better | Good | TextBlob | Better polarity shifts for phrases like "not bad" |
| Context understanding | Weak | Weak | Strong | Transformer | Learns contextual meaning |

---

## Key Findings Summary
- VADER achieved the highest raw accuracy
- TextBlob handled polarity shifts better than VADER
- The Transformer best understood contextual meaning
- Accuracy alone did not reflect true language understanding
- Complex language separated model performance more than simple sentences

Overall:  
The Transformer most closely matched human reasoning in nuanced cases even though it had the lowest accuracy score.
