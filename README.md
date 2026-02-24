# Prompt Injection Detection for LLM Safety

## Table of Contents
- [Overview](#overview)
- [Why this problem matters](#why-this-problem-matters)
- [Data](#data)
- [Implementation Details](#implementation-details)
- [Key results](#key-results)
- [How to run](#how-to-run)
- [Repository structure](#repository-structure)
- [Limitations and next improvements](#limitations-and-next-improvements)

## Overview
This project builds a prompt-injection detector that classifies prompts as benign (`0`) or injection-like (`1`). I trained and compared:
- A simple baseline (character TF-IDF + Logistic Regression)
- A fine-tuned transformer (`distilbert-base-uncased`)

The end goal is a practical guardrail decision (`ALLOW` or `BLOCK`) with a tunable probability threshold, exposed in a Streamlit app.

## Why this problem matters
Prompt injection is one of the most common real-world failure modes for LLM applications. A guardrail model should prioritize not letting malicious prompts slip through. That means false negatives (injection predicted as benign) are usually more dangerous than false positives (benign predicted as injection).

## Data
### In-domain training data
- Source: `S-Labs/prompt-injection-dataset` (Hugging Face)
- Label meaning: `0 = benign`, `1 = injection`
- Split sizes used in the notebook:
  - Train: 11,089
  - Validation: 2,101
  - Test: 2,101

### External generalization data
- Source: `deepset/prompt-injections` (Hugging Face)
- Used only for out-of-domain evaluation (not training), to test how well each model generalizes.

## Implementation Details
This is the exact workflow implemented in `notebooks/main_analysis.ipynb`.

### 1) Load and prepare the dataset
- Load dataset from Hugging Face.
- Convert to pandas for easier inspection and analysis.
- Keep the task as standard binary classification where label `1` is the positive (injection) class.

Why: a consistent label mapping is critical for interpreting precision/recall correctly.

### 2) Create train/val/test splits
- Use the provided split sizes (train, validation, test).

Why: validation is needed for tuning training decisions, and the test set is held out for final reporting.

### 3) Baseline model: character TF-IDF + Logistic Regression
- Feature extractor: character-level TF-IDF with `ngram_range=(3,5)`
- Classifier: Logistic Regression

Why this baseline:
- Character n-grams can catch patterns like obfuscation, weird spacing, punctuation tricks, and common injection phrases.
- Logistic Regression is fast, strong on many text classification problems, and a good sanity check.

### 4) Transformer model: fine-tuned DistilBERT
- Model: `distilbert-base-uncased` fine-tuned for binary classification.
- Tokenization: truncation enabled, max length set to 256.

Why this model:
- DistilBERT is a strong general-purpose encoder that often generalizes better under domain shift than sparse bag-of-ngrams.
- The tradeoff is higher compute and sometimes more aggressive behavior on short or ambiguous text.

### 5) Evaluation metrics and analysis

I reported:
- Accuracy, precision, recall, F1 (focused on class `1`, injection)
- Confusion matrices
- Error analysis and qualitative examples

Why these metrics:
- Accuracy can look good even when the model misses many injections.
- Recall on class `1` matters a lot for safety because false negatives mean an injection gets allowed.

### 6) External generalization check
- Run both models on `deepset/prompt-injections`.

Why: In-domain scores can be misleading. External testing shows whether the model learned general patterns or just dataset-specific cues.

### 7) Guardrail deployment (Streamlit app)
- Load saved artifacts:
  - Baseline: `tfidf.joblib`, `logreg.joblib`
  - Transformer: fine-tuned model folder
- Compute `P(injection)` and compare to a threshold to output `ALLOW` or `BLOCK`.

Why: This matches real app workflows where you want a single decision and a confidence score, not just a label.

## Key results
### In-domain test set (`S-Labs/prompt-injection-dataset`)
- Baseline (TF-IDF + LR):
  - Accuracy: `0.9357`
  - Precision (class 1): `0.9935`
  - Recall (class 1): `0.8773`
  - F1 (class 1): `0.9318`

- Transformer (DistilBERT):
  - Accuracy: `0.9604`
  - Precision (class 1): `0.9859`
  - Recall (class 1): `0.9343`
  - F1 (class 1): `0.9594`

Interpretation:
- TF-IDF is a strong baseline for this task, especially for precision.
- The baseline misses more injections (lower recall), which is risky for guardrails.
- The transformer is more balanced and catches more injections (higher recall), with a small precision drop.

### External dataset (`deepset/prompt-injections`)
- Baseline:
  - Accuracy: `0.7509`
  - Injection F1: `0.5000`
  - Injection recall: `0.3350` (very low)

- Transformer:
  - Accuracy: `0.8022`
  - Injection F1: `0.7173`
  - Injection recall: `0.6749`
  - Injection precision: `0.7654`

Interpretation:
- The baseline collapses on recall under domain shift, it often predicts benign even when the prompt is injection.
- The transformer generalizes better for safety, it reduces false negatives even if it blocks some benign prompts.


### Guardrail behavior notes (qualitative)
From the notebook demo examples, injection-style prompts were high-confidence `BLOCK` (around `0.9`) while normal requests like summarization and translation were typically `ALLOW`.

I also observed a consistent edge case during manual testing:
- Very short, ambiguous prompts (examples: "hi", "hello") can cause the transformer to behave more aggressively and sometimes over-predict injection.
- The baseline is often more stable on these tiny inputs.

This is a known practical tradeoff: the transformer is better for catching attacks, but it may over-block low-information prompts.

Note: If you retrain the transformer, you should expect slightly different results each run.


## How to run
1) Create and activate a virtual environment (Python 3.10+).
2) Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook to reproduce training and evaluation:

   * Open `notebooks/main_analysis.ipynb`
   * Run all cells
   * This will also save model artifacts under `outputs/models/`
4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Repository structure

```
.
├── README.md
├── requirements.txt
├── app.py
├── data/
│   └── processed/
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
├── notebooks/
│   └── main_analysis.ipynb
└── outputs/
    ├── figures/
    │   ├── baseline_confusion_matrix.png
    │   ├── transformer_confusion_matrix.png
    │   ├── model_scoreboard.png
    │   ├── guardrail_risk_meter.png
    │   └── train_text_length_hist.png
    └── models/
        ├── baseline_tfidf_logreg/
        ├── distilbert_prompt_injection_final/
        └── distilbert_prompt_injection_run/
```

## Limitations and next improvements

* Data mismatch: external datasets can differ in style and label definitions, so any single score is not the whole story.
* Short prompts: add a simple rule like "if length < N characters then lower confidence" or treat as "unknown" unless other signals exist.
* Broader evaluation: test on multilingual prompts, obfuscated attacks, and prompts that contain both benign instructions and injection segments.




