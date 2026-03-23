# AI Usage Log — Assignment 2: Text Classification
**Student:** Josh  
**Course:** BSAN 6080  
**Date:** March 2026

---

## Overview
AI was used as a coding assistant for implementation, debugging, and structuring 
code throughout this assignment. All error analysis, model selection reasoning, 
manual review of the 20 test rows, and inference reflection were completed 
independently without AI assistance. This log documents how AI was used, what 
I changed, and what I learned from each interaction.

---

## Learning Progression from A1 to A2

### A1 Prompts (vague, one-line):
> "Help me clean the text for nlp and handle the missing values"  
> "apply VADER sentiment to the text column and create a column for the compound score"  
> "compute sentiment polarity for textblob model"

### A2 Prompts (specific, context-rich):
> "I have a fake news binary classification task using TF-IDF with 5000 features 
> and ngram range 1-2. My models are LR, SVM, and Naive Bayes trained with 
> class_weight balanced. The test F1 is very high but my custom inference on 
> 20 new examples only got 65%. What does this tell me about generalization?"

> "I need GridSearchCV on Logistic Regression for binary text classification 
> with TF-IDF features. Scoring should be f1_weighted, use 5-fold CV, test C 
> values of 0.01 to 100 and both lbfgs and saga solvers."

**What changed:**  
In A1 I gave the AI no context about my dataset, task, or existing code and just 
described what I wanted in one line. By A2 I learned to include the specific task 
type, variable names, what I had already tried, and what metric I was optimizing 
for. This produced more accurate responses with less back and forth debugging.

---

## Task 1: Combining and Labeling Data
**What I was trying to do:**  
Add labels to each dataframe and combine them into one shuffled dataset.

**AI Prompt:**  
"give me the code to add 1 for the true and 0 for the fake then how to combine them"

**AI Response Used:**  
```python
df_true["label"] = 1
df_fake["label"] = 0
df = pd.concat([df_true, df_fake], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

**What I Modified:**  
Nothing — this matched exactly what I needed.

**What I Learned:**  
sample(frac=1) shuffles all rows and reset_index(drop=True) prevents the old 
index from becoming a column. Both are necessary after concatenating two ordered 
dataframes.

---

## Task 2: Fixing Shape Error
**What I was trying to do:**  
Print the shape of the dataframes after loading.

**My Analysis Before AI:**  
I noticed the error said "tuple object is not callable" which told me I was 
treating shape like a function. I recognized shape is a property not a method 
so parentheses were not needed.

**AI Prompt:**  
Confirmed fix with Claude after identifying the issue myself.

**What I Modified:**  
Removed parentheses from df.shape() → df.shape

**What I Learned:**  
shape, dtypes, columns, and index are all properties in pandas — they do not 
use parentheses. Methods like head(), describe(), and info() do.

---

## Task 3: Text Cleaning Pipeline
**What I was trying to do:**  
Build a domain-appropriate cleaning function for formal news text.

**AI Prompt:**  
"design domain appropriate cleaning for news and formal text, show before and 
after examples for each class"

**AI Response Used:**  
Provided a cleaning function covering lowercase, URL removal, email removal, 
number removal, punctuation removal, stopword filtering, lemmatization, and 
short token removal.

**What I Modified:**  
Adjusted stopword import to match my existing nltk setup from the professor's 
example. Fixed a case comparison issue where tokens were not lowercased before 
being checked against the stopword set.

**What I Learned:**  
Stopword filtering only works correctly if tokens are lowercased first. 
News text cleaning should remove numbers and punctuation since formal news 
does not rely on emphasis signals the way sentiment text does.

---

## Task 4: Feature Engineering Comparison
**What I was trying to do:**  
Compare TF-IDF and Count Vectorizer to justify my feature choice.

**AI Prompt:**  
"show me step 4 tokenize and feature engineer, i need to try at least 2 approaches 
and compare them, vocabulary size for binary is 1000 to 5000"

**AI Response Used:**  
Provided TF-IDF and Count Vectorizer code with matching parameters and a 
Logistic Regression comparison to show F1 difference between the two approaches.

**What I Modified:**  
Nothing significant — parameters already matched my task requirements.

**What I Learned:**  
TF-IDF outperforms Count Vectorizer on news text because it down-weights 
high frequency filler words like "said" and "according" that appear in both 
classes and carry no discriminating signal.

---

## Task 5: Model Training
**What I was trying to do:**  
Train Logistic Regression, Linear SVM, and Naive Bayes on TF-IDF features 
with class imbalance handling.

**AI Prompt:**  
"show me step 5 train models, this is just binary classification, i need at 
least 2 algorithms with class weight balanced"

**AI Response Used:**  
Provided training loop storing predictions and F1 scores in a results dictionary 
for all three models.

**What I Modified:**  
Kept class_weight='balanced' on LR and SVM. Noted Naive Bayes does not support 
this parameter directly.

**What I Learned:**  
Storing results in a dictionary keyed by model name makes evaluation and 
comparison cleaner. LinearSVC does not have predict_proba so confidence scores 
require a different approach.

---

## Task 6: Hyperparameter Tuning
**What I was trying to do:**  
Tune Logistic Regression before asking for help I tried setting up a small 
manual grid testing C values of 0.1, 1, and 10.

**My Analysis Before AI:**  
I understood C controls regularization strength and that lower C means stronger 
regularization. My manual test showed C=1 performing best in my small grid so 
I asked Claude to help expand this properly with cross validation.

**AI Prompt:**  
"I need GridSearchCV on Logistic Regression for binary text classification with 
TF-IDF features, scoring f1_weighted, 5-fold CV, test C values 0.01 to 100 
and both lbfgs and saga solvers"

**AI Response Used:**  
Provided full GridSearchCV setup with param_grid, cv=5, n_jobs=-1, and storing 
the tuned model back into the results dictionary.

**What I Modified:**  
Added the tuned model to results dict so it appeared in all downstream 
evaluation and comparison cells automatically.

**What I Learned:**  
GridSearchCV with n_jobs=-1 uses all CPU cores which significantly speeds up 
the search. The tuned model improved F1 from 0.9919 to 0.9976 on the Fake class.

---

## Task 7: Identifying High F1 as a Concern
**What I was trying to do:**  
Understand why test F1 was near perfect.

**My Analysis Before AI:**  
I flagged this independently before asking Claude. I reasoned that real and fake 
articles came from completely different sources and that TF-IDF was probably picking up on vocabulary style patterns 
rather than actual misinformation signals. This was confirmed when the same model 
dropped to 65% on 20 fresh AI-generated examples.

**AI Prompt:**  
"i feel like my training recall and precision is too high"

**AI Response Used:**  
Claude confirmed the source-level leakage concern and explained it as a known 
criticism of this specific dataset.

**What I Learned:**  
Near-perfect accuracy on a train-test split from the same dataset does not 
mean the model will generalize. Testing on truly new out-of-distribution 
examples is the real test of generalization.

---

## When I Beat AI

### Case 1: Catching Source-Level Leakage Before AI
I independently noticed and questioned the near-perfect F1 scores before asking 
Claude about it. Claude's initial response treated the high scores as a success. 
My skepticism led to the most important finding of the assignment — that the model 
learned writing style not misinformation signals — which became the key limitation 
in the technical memo.

### Case 2: Important Class Metric — Pushing Back on Recall
Claude initially recommended optimizing for Recall on the Fake class using the 
spam detection analogy. I pushed back after re-reading the rubric which explicitly 
listed "Fake news → optimize balance" as the example. I argued that suppressing 
real journalism is not a minor recoverable error the way spam in an inbox is. 
The rubric confirmed my reasoning was correct and weighted F1 became the 
justified primary metric.
