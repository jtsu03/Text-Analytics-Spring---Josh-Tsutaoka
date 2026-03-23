# AI Usage Log — Sentiment Analysis Comparison

This document records how AI was used during the development of this project.  
For each interaction, I include the goal, the prompt given to the AI, the code that was used, what I modified, and what I learned.

The AI was used primarily as a coding assistant to:
- help generate initial implementations
- troubleshoot errors
- structure evaluation methods
- speed up repetitive programming tasks

However, model design decisions, preprocessing logic, interpretation of results, and comparison conclusions were determined manually.

The project compares three sentiment approaches:
- VADER (rule-based sentiment)
- TextBlob (lexicon polarity)
- Transformer (context-based emotion model)

The objective is to evaluate how different NLP approaches behave on real review text and understand why models agree or disagree, not just which has the highest accuracy.

## Task 1: Cleaning Review Text for Sentiment Models

**What I was trying to do:**  
Prepare raw reviews so sentiment models would not fail on HTML, symbols, or missing values.

**AI Prompt:**  
Help me clean the text for nlp and handle the missing values

**AI Response (used):**
```python
def clean_review(t):
    t = "" if pd.isna(t) else str(t)
    t = html.unescape(t)
    t = re.sub(r"<br\s*/?>", " ", t)
    t = re.sub(r"[^A-Za-z\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    if t == "":
        return "emptyreview"
    return t.lower()

data["text"] = data["review"].apply(clean_review)
```

**What I Modified:**  
Added the `"emptyreview"` placeholder so VADER would not crash on blank strings.

**What I Learned:**  
- Text cleaning must handle NaN values before NLP processing  
- Sentiment tools fail on empty input  
- Lowercasing and removing symbols improves consistency  

**AI Errors Found:**  
Returning an empty string caused scoring errors, so a placeholder was required.

## Task 2: Calculating VADER Compound Sentiment Score

**What I was trying to do:**  
Compute a sentiment score for each review using VADER.

**AI Prompt:**  
apply VADER sentiment to the text column and create a column for the compound score

**AI Response (used):**
```python
data["vader_compound"] = data["text"].apply(lambda t: sid.polarity_scores(t)["compound"])
data[["text", "vader_compound"]].head(5)
```

**What I Modified:**  
Used `.apply()` so the analyzer runs on each review and stored only the compound score.

**What I Learned:**  
- VADER returns multiple values, but compound is the overall sentiment score  
- Scores range from -1 (negative) to +1 (positive)  
- Pandas `.apply()` is the simplest way to run NLP functions across text data

## Task 3: Converting VADER Scores to Sentiment Labels

**What I was trying to do:**  
Turn the VADER compound score into categorical sentiment (positive, negative, neutral).

**AI Prompt:**  
convert VADER compound scores into classes neg, neu, and pos with positive threshold of 0.05 and a negative threshold of -0.05

**AI Response (used):**
```python
def vader_label(c, pos_thr=0.05, neg_thr=-0.05):
    if c >= pos_thr:
        return "pos"
    elif c <= neg_thr:
        return "neg"
    else:
        return "neu"

data["vader_label"] = data["vader_compound"].apply(vader_label)
data["vader_label"].value_counts()
```

**What I Modified:**  
Used the standard ±0.05 thresholds and applied them across all rows.

**What I Learned:**  
- VADER outputs continuous scores that must be categorized for comparison  
- Threshold choice determines how many neutral predictions appear  
- Counting labels helps verify distribution before evaluation

## Task 4: Calculating TextBlob Polarity Score

**What I was trying to do:**  
Generate a sentiment score for each cleaned review using TextBlob.

**AI Prompt:**  
compute sentiment polarity for textblob model

**AI Response (used):**
```python
data["tb_polarity"] = data["text"].apply(lambda t: TextBlob(t).sentiment.polarity)

data[["text","tb_polarity"]].head(5)
```

**What I Modified:**  
Applied the function directly to the cleaned text column so results match VADER input.

**What I Learned:**  
- TextBlob polarity ranges from -1 to +1  
- Applying it after cleaning prevents errors and inconsistent scores  
- Different sentiment tools produce different scoring behavior

## Task 5: Converting TextBlob Polarity to Sentiment Labels

**What I was trying to do:**  
Convert TextBlob polarity scores into positive, negative, or neutral categories for comparison with VADER and the transformer model.

**AI Prompt:**  
turn TextBlob polarity into classes neg, neu, and pos with thresholds of 0.05 for positive and -0.05 for negative

**AI Response (used):**
```python
def tb_label(p, pos_thr=0.05, neg_thr=-0.05):
    if p >= pos_thr:
        return "pos"
    elif p <= neg_thr:
        return "neg"
    else:
        return "neu"

data["tb_label"] = data["tb_polarity"].apply(tb_label)

data["tb_label"].value_counts()
```

**What I Modified:**  
Used the same ±0.05 thresholds as VADER so model outputs are comparable.

**What I Learned:**  
- Consistent thresholds are needed when comparing models  
- Polarity scores must be categorized before evaluation  
- Label counts help verify sentiment distribution

## Task 6: Running Transformer Emotion Classifier

**What I was trying to do:**  
Apply a pretrained transformer model to every review and store the predicted emotion and confidence score.

**AI Prompt:**  
run a Hugging Face text-classification pipeline for the text column

**AI Response (used):**
```python
emo_clf = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=0 if torch.cuda.is_available() else -1
)

texts = data["text"].fillna("").astype(str).tolist()

hf_emotion = []
hf_emotion_score = []

batch_size = 32
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    preds = emo_clf(batch, truncation=True)
    hf_emotion.extend([p["label"] for p in preds])
    hf_emotion_score.extend([p["score"] for p in preds])

data["hf_emotion"] = hf_emotion
data["hf_emotion_score"] = hf_emotion_score
```

**What I Modified:**  
Ensured text had no missing values and stored predictions in lists before assigning back to the dataframe.

**What I Learned:**  
- Transformers should process text in batches for speed  
- The pipeline returns both a label and confidence score  
- GPU detection allows the same code to run faster when available  

**AI Errors Found:**  
Running predictions one review at a time was too slow, so batching was required.

## Task 7: Converting Transformer Emotions to Sentiment Labels

**What I was trying to do:**  
Map the transformer’s emotion predictions into positive or negative sentiment so they could be compared with other models.

**AI Prompt:**  
convert emotion classification results into sentiment labels

**AI Response (used):**
```python
def emotion_to_binary(e):
    e = str(e).lower().strip()
    if e in ["joy", "surprise"]:
        return "pos"
    else:
        return "neg"

data["hf_label"] = data["hf_emotion"].apply(emotion_to_binary)
```

**What I Modified:**  
Standardized the emotion text (lowercase + strip) before mapping.

**What I Learned:**  
- Emotion outputs must be converted to a common label system for comparison  
- Cleaning labels prevents mismatches during evaluation  
- Mapping decisions affect final model accuracy

## Task 8: Creating Final Comparison Dataset

**What I was trying to do:**  
Combine all model outputs into one dataframe to make comparison and evaluation easier.

**AI Prompt:**  
Create a final dataset with these columns

**AI Response (used):**
```python
final_df = data[[
    "drugName",
    "text",

    "vader_label",
    "vader_compound",

    "tb_label",
    "tb_polarity",

    "hf_label",
    "hf_emotion",
    "hf_emotion_score",
  
]].copy()
```

**What I Modified:**  
Used `.copy()` to avoid modifying the original dataframe during later analysis.

**What I Learned:**  
- Keeping results in a single table simplifies model comparison  
- `.copy()` prevents unintended changes to the source data  
- Organizing outputs is important before evaluation

## Task 9: Measuring Model Disagreement

**What I was trying to do:**  
Quantify how different the models’ predictions were from each other.

**AI Prompt:**  
Show the differences in the models, keep each label for all the models

**AI Response (used):**
```python
tmp = final_df.copy()

tmp["vader_tb_gap"] = (tmp["vader_compound"] - tmp["tb_polarity"]).abs()
tmp["vader_hf_gap"] = (tmp["vader_compound"] - tmp["hf_emotion_score"]).abs()
tmp["tb_hf_gap"] = (tmp["tb_polarity"] - tmp["hf_emotion_score"]).abs()

tmp[["vader_tb_gap","vader_hf_gap","tb_hf_gap"]].describe()
```

**What I Modified:**  
Used absolute differences so gaps represent disagreement magnitude instead of direction.

**What I Learned:**  
- Comparing models requires numeric distance, not just labels  
- `.describe()` helps summarize overall agreement patterns  
- Different sentiment methods can vary significantly in confidence

## Task 10: Inspecting Reviews Where Models Disagree

**What I was trying to do:**  
Find reviews where the transformer prediction differed from both VADER and TextBlob to analyze model behavior.

**AI Prompt:**  
filter rows where the models predicted different values between transformer vs VADER and Textblob

**AI Response (used):**
```python
final_df[
    (final_df["hf_label"] != final_df["vader_label"]) &
    (final_df["hf_label"] != final_df["tb_label"])
][
    ["drugName","text","hf_emotion","hf_label","hf_emotion_score",
     "vader_label","vader_compound","tb_label","tb_polarity"]
].head(10)
```

**What I Modified:**  
Selected relevant columns only so disagreements are easier to interpret.

**What I Learned:**  
- Filtering helps understand model differences beyond accuracy  
- Transformers often disagree with lexicon models on contextual language  
- Inspecting examples is necessary to interpret results

## Task 11: Creating a Random Sample for Manual Review

**What I was trying to do:**  
Select a manageable subset of reviews for inspection and evaluation.

**AI Prompt:**  
I want to get a sample of 100 random to be able to export for human labels

**AI Response (used):**
```python
sample100_final = final_df.sample(n=100, random_state=42).copy()

sample100_final.head(10)
```

**What I Modified:**  
Used a fixed `random_state` so results are reproducible.

**What I Learned:**  
- Sampling helps manual analysis without processing the full dataset  
- Reproducibility is important when sharing results  
- `.copy()` prevents accidental modification of the original data

## Task 12: Preparing Labels for Evaluation Metrics

**What I was trying to do:**  
Make sure all model labels and human labels matched format before computing accuracy and confusion matrices.

**AI Prompt:**  
standardize labels before doing metrics, create confusion matrix, and create classification report

**AI Response (used):**
```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

for col in ["human_label","vader_label","tb_label","hf_label"]:
    final[col] = final[col].astype(str).str.lower().str.strip()
```

**What I Modified:**  
Applied the same formatting to every label column instead of fixing them individually.

**What I Learned:**  
- Evaluation metrics require identical label formatting  
- Extra spaces or capitalization can produce incorrect results  
- Cleaning labels is necessary before computing accuracy

## Task 13: Creating a Reusable Evaluation Function

**What I was trying to do:**  
Build a function to consistently evaluate each model using confusion matrix, accuracy, and classification report.

**AI Prompt:**  
create a confusion matrix and classification report in sklearn?

**AI Response (used):**
```python
def evaluate(name, y_true, y_pred):

    print(f"\n===== {name} =====")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm,
                         index=[f"True_{l}" for l in labels],
                         columns=[f"Pred_{l}" for l in labels])
    display(cm_df)

    print("Accuracy:", round(accuracy_score(y_true, y_pred),4))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
```

**What I Modified:**  
Added `labels=labels` and `zero_division=0` to keep outputs consistent and avoid warnings.

**What I Learned:**  
- Wrapping metrics in a function ensures fair comparison across models  
- Confusion matrices are easier to read as dataframes  
- Handling zero divisions prevents evaluation errors  

**AI Errors Found:**  
Default sklearn output sometimes warns on missing classes, so parameters were needed.

## Task 14: Evaluating All Models Against Human Labels

**What I was trying to do:**  
Run the same evaluation metrics for VADER, TextBlob, and the transformer model to compare performance.

**AI Prompt:**  
Give me code to evaluate the multiple models

**AI Response (used):**
```python
evaluate("VADER", final["human_label"], final["vader_label"])
evaluate("TextBlob", final["human_label"], final["tb_label"])
evaluate("Transformer", final["human_label"], final["hf_label"])
```

**What I Modified:**  
Used a reusable evaluation function so each model is measured consistently.

**What I Learned:**  
- Consistent evaluation is necessary for fair comparison  
- Human labels act as the ground truth  
- Running models side-by-side highlights performance differences

## Task 15: Summarizing Model Accuracy

**What I was trying to do:**  
Create a table comparing the accuracy of VADER, TextBlob, and the transformer model.

**AI Prompt:**  
build a dataframe comparing accuracy scores of the models?

**AI Response (used):**
```python
summary = pd.DataFrame({
    "Model":["VADER","TextBlob","Transformer"],
    "Accuracy":[
        accuracy_score(final["human_label"], final["vader_label"]),
        accuracy_score(final["human_label"], final["tb_label"]),
        accuracy_score(final["human_label"], final["hf_label"])
    ]
}).sort_values("Accuracy", ascending=False)

summary
```

**What I Modified:**  
Sorted by accuracy to immediately see the best performing model.

**What I Learned:**  
- Summarizing results helps interpret evaluation quickly  
- Accuracy comparison shows relative model performance  
- Presenting results clearly is part of analysis

## Task 16: Marking Correct Predictions

**What I was trying to do:**  
Identify which predictions from each model matched the human label.

**AI Prompt:**  
compare two models and show me if they match?

**AI Response (used):**
```python
final["vader_correct"] = final["vader_label"] == final["human_label"]
final["tb_correct"] = final["tb_label"] == final["human_label"]
final["hf_correct"] = final["hf_label"] == final["human_label"]
```

**What I Modified:**  
Created a separate column for each model so correctness could be analyzed individually.

**What I Learned:**  
- Boolean comparisons can track model performance per row  
- This allows deeper analysis beyond overall accuracy  
- Helpful for inspecting specific successes and failures

## Task 17: Counting Correct vs Incorrect Predictions and Calculating Accuracy

**What I was trying to do:**  
Summarize how many predictions each model got right and wrong and compute accuracy from those counts.

**AI Prompt:**  
Summarize the predictions each model got right and wrong

**AI Response (used):**
```python
correct_counts = pd.DataFrame({
    "Model": ["VADER", "TextBlob", "Transformer"],
    "Correct": [
        final["vader_correct"].sum(),
        final["tb_correct"].sum(),
        final["hf_correct"].sum()
    ],
    "Incorrect": [
        (~final["vader_correct"]).sum(),
        (~final["tb_correct"]).sum(),
        (~final["hf_correct"]).sum()
    ]
})

correct_counts
```

**What I Modified:**  
Used boolean sums and negation (`~`) to calculate both correct and incorrect counts.

**What I Learned:**  
- True values count as 1 in pandas, allowing quick totals  
- Breaking accuracy into counts gives clearer interpretation  
- Useful for comparing models beyond percentages  

## Task 18: Calculating Accuracy from Prediction Counts

**AI Prompt:**  
calculate accuracy from correct and incorrect

**AI Response (used):**
```python
correct_counts["Accuracy"] = correct_counts["Correct"] / (correct_counts["Correct"] + correct_counts["Incorrect"])
correct_counts.sort_values("Accuracy", ascending=False)
```

**What I Modified:**  
Sorted results so the best performing model appears first.

**What I Learned:**  
- Accuracy is the proportion of correct predictions  
- Calculating it manually verifies sklearn results  
- Sorting helps quickly interpret performance

## Task 19: Displaying Correct Prediction Examples

**What I was trying to do:**  
Show sample reviews where a model correctly matched the human label.

**AI Prompt:**  
Get a sample of reviews where the model was correct

**AI Response (used):**
```python
def show_success_examples(df, model_label, score_col, model_name, n=3):

    success = df[df[model_label] == df["human_label"]].sample(n, random_state=42)

    for i, row in enumerate(success.itertuples(), 1):
        print(f"\n{model_name} Success Example {i}:")
        print("Text:", row.text[:300], "...")
        print(f"{model_name} prediction:", row.__getattribute__(model_label),
              f"({round(row.__getattribute__(score_col),3)})")
        print("Human label:", row.human_label)
```

**What I Modified:**  
Limited text length and fixed the random seed so examples are readable and reproducible.

**What I Learned:**  
- Sampling helps illustrate model behavior clearly  
- Showing confidence scores adds interpretation context  
- Example inspection complements quantitative metrics

## Task 20: Viewing Correct Predictions

**What I was trying to do:**  
Display sample reviews where VADER matched the human label.

**AI Prompt:**  
Show the reviews where models matched the human label

**AI Response (used):**
```python
show_success_examples(final, "vader_label", "vader_compound", "VADER")
```

**What I Modified:**  
Passed the VADER label and score columns so the function prints the correct model output. Modified this for every pretrained model as well.

**What I Learned:**  
- Functions can be reused for different models by passing column names  
- Viewing correct cases helps understand model strengths  
- Qualitative inspection supports metric results

## Task 21: Displaying Incorrect Prediction Examples

**What I was trying to do:**  
Show sample reviews where a model prediction did not match the human label.

**AI Prompt:**  
Sample reviews where models predictions aren't matching human label

**AI Response (used):**
```python
def show_failure_examples(df, model_label, score_col, model_name, n=3):

    failures = df[df[model_label] != df["human_label"]].sample(n, random_state=42)

    for i, row in enumerate(failures.itertuples(), 1):
        print(f"\n{model_name} Failure Example {i}:")
        print("Text:", row.text[:300], "...")
        print(f"{model_name} prediction:", row.__getattribute__(model_label),
              f"({round(row.__getattribute__(score_col),3)})")
        print("Actual sentiment:", row.human_label)
```

**What I Modified:**  
Limited text length and fixed the random seed so examples are readable and reproducible.

**What I Learned:**  
- Inspecting mistakes explains model weaknesses  
- Comparing prediction vs actual label helps interpret errors  
- Qualitative examples complement accuracy metrics

## Task 22: Viewing Incorrect VADER Predictions

**What I was trying to do:**  
Display sample reviews where VADER did not match the human label.

**AI Prompt:**  
Show failure cases for all models

**AI Response (used):**
```python
show_failure_examples(final, "vader_label", "vader_compound", "VADER")
```

**What I Modified:**  
Passed the VADER label and score columns so the function shows the correct comparison.

**What I Learned:**  
- Reviewing errors helps understand model limitations  
- Functions allow repeated analysis across models  
- Qualitative review supports quantitative results

## Task 23: Build Model Comparison Table

**What I was trying to do:**  
Create a table comparing VADER, TextBlob, and Transformer models across multiple criteria (speed, accuracy, negation handling, emphasis handling, and contextual understanding). The goal was to calculate speed and accuracy directly from notebook outputs instead of manually entering values.

**AI Prompt:**  
Give me Python code to build a comparison table using my dataframe with vader_label, tb_label, hf_label, and human_label. Also benchmark model speed and calculate accuracy.

**AI Response (used):**
```python
def sample_texts(df, text_col="text", n=500, seed=42):
    s = df[text_col].fillna("").astype(str)
    n = min(n, len(s))
    return s.sample(n, random_state=seed).tolist()

def model_accuracy(df, pred_col, truth_col="human_label"):
    d = df.dropna(subset=[pred_col, truth_col]).copy()
    return accuracy_score(d[truth_col], d[pred_col])

def vader_speed(texts, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = [sid.polarity_scores(t)["compound"] for t in texts]
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times))

def textblob_speed(texts, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = [TextBlob(t).sentiment.polarity for t in texts]
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times))

def transformer_speed(texts, batch_size=32, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            _ = emo_clf(texts[i:i+batch_size], truncation=True)
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times))

# Run benchmarks
N_SPEED = 500
texts = sample_texts(final, n=N_SPEED)

vader_sec = vader_speed(texts)
tb_sec    = textblob_speed(texts)
hf_sec    = transformer_speed(texts)

vader_acc = model_accuracy(final, "vader_label")
tb_acc    = model_accuracy(final, "tb_label")
hf_acc    = model_accuracy(final, "hf_label")

def winner_speed(v, t, h):
    d = {"VADER": v, "TextBlob": t, "Transformer": h}
    return min(d, key=d.get)

def winner_accuracy(v, t, h):
    d = {"VADER": v, "TextBlob": t, "Transformer": h}
    return max(d, key=d.get)

speed_winner = winner_speed(vader_sec, tb_sec, hf_sec)
acc_winner   = winner_accuracy(vader_acc, tb_acc, hf_acc)

def fmt_sec(x): return f"{x:.2f} sec / {N_SPEED} reviews"
def fmt_pct(x): return f"{x*100:.1f}%"

rows = [
    {
        "Criterion": "Speed",
        "VADER": fmt_sec(vader_sec),
        "TextBlob": fmt_sec(tb_sec),
        "Transformer": fmt_sec(hf_sec),
        "Winner": speed_winner,
        "Justification": "Lower runtime is faster"
    },
    {
        "Criterion": "Accuracy (human labels)",
        "VADER": fmt_pct(vader_acc),
        "TextBlob": fmt_pct(tb_acc),
        "Transformer": fmt_pct(hf_acc),
        "Winner": acc_winner,
        "Justification": "Highest agreement with human labels"
    },
    {
        "Criterion": "Handles emphasis (caps/punctuation)",
        "VADER": "Excellent",
        "TextBlob": "Weak",
        "Transformer": "Good",
        "Winner": "VADER",
        "Justification": "Rule based intensity handling"
    },
    {
        "Criterion": "Handles negation",
        "VADER": "Good",
        "TextBlob": "Better",
        "Transformer": "Good",
        "Winner": "TextBlob",
        "Justification": "Better polarity shifts for phrases like 'not bad'"
    },
    {
        "Criterion": "Context understanding",
        "VADER": "Weak",
        "TextBlob": "Weak",
        "Transformer": "Strong",
        "Winner": "Transformer",
        "Justification": "Learns contextual meaning"
    },
]

comparison = pd.DataFrame(rows)
comparison
```

**How I used the response:**  
Adapted the code to match my variable names and models already defined in the notebook (sid for VADER, TextBlob, and the transformer pipeline). I separated the logic into multiple cells and added comments to explain each step.

**What I changed:**  
- Matched the column names to my dataset  
- Used stored predictions instead of recomputing them  
- Added qualitative comparison criteria and justifications  
- Formatted the output for clean notebook display  

**Limitations / Issues:**  
The original suggestion recomputed predictions, which would have been inefficient, so I modified the code to use existing predictions instead.

**What I learned:**  
How to systematically compare NLP models using both quantitative metrics (accuracy and speed) and qualitative linguistic evaluation.

**What worked well:**  
The runtime benchmarking and automatic winner selection worked correctly and prevented manual calculation errors.
