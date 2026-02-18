# Sentiment Analysis Model Comparison
**VADER vs TextBlob vs Transformer**

## Project Summary
This project compares three sentiment analysis approaches on real user review text and evaluates how closely each model matches human interpretation.

Rather than focusing only on accuracy, the goal is to understand the tradeoffs between speed, linguistic understanding, and contextual reasoning across different generations of NLP models.

Models compared:
- **VADER** — rule-based lexicon model
- **TextBlob** — traditional polarity scoring model
- **Transformer (HuggingFace)** — pretrained deep learning model

---

## Dataset
The dataset consists of natural language user reviews containing opinions about a product or experience. The text is informal and varies significantly in writing style, making it useful for evaluating real-world sentiment interpretation.

Challenges present in the data:
- Negation ("not bad")
- Emphasis (capitalization and punctuation)
- Mixed sentiment ("worked but caused side effects")
- Context-dependent meaning

A subset of reviews was manually labeled to create a ground-truth benchmark.

**Sentiment labels**
- Positive
- Neutral
- Negative

---

## Methodology

### 1. Preprocessing
Text was lightly cleaned while preserving emotional cues:
- Removed URLs and noise characters
- Normalized spacing
- Lowercased text
- Preserved punctuation important for sentiment

### 2. Model Predictions
Each model generated sentiment labels for the same reviews.

**VADER**
- Uses a sentiment dictionary and linguistic rules
- Accounts for punctuation, capitalization, and negation

**TextBlob**
- Calculates polarity based on learned word patterns

**Transformer**
- Uses contextual embeddings to interpret full sentence meaning

### 3. Evaluation
Models were compared against human labels using:
- Accuracy
- Agreement analysis
- Example successes
- Example failures

---

## Results

| Criterion | VADER | TextBlob | Transformer |
|----------|------|------|------|
| Speed | Fastest | Moderate | Slowest |
| Accuracy | Good | Moderate | Highest |
| Emphasis handling | Strong | Weak | Good |
| Negation handling | Moderate | Better | Good |
| Context understanding | Limited | Limited | Strong |

---

## Key Takeaways
- **VADER** performs well when speed is required and text relies heavily on punctuation.
- **TextBlob** handles simple negation better but lacks contextual awareness.
- **Transformer** produces the most human-like predictions but requires significantly more computation.

There is no universally best model — the correct choice depends on whether performance or interpretability is more important.

---

## Technologies
Python, Pandas, NLTK, TextBlob, HuggingFace Transformers, Scikit-learn

---

## How to Run
1. Open `Text_Mining_Sentiment_Analysis_Assignment_1.ipynb`
2. Run all cells sequentially
3. View the comparison table and error analysis
