# NLP Assignment 04 — PDF-Based NLP Pipeline
### Next Word Prediction and Text Classification

**Course:** Natural Language Processing Lab
**University:** Shifa Tameer-e-Millat University, Islamabad
**Instructor:** Mr. Sultan Haider
**Student:** [apna naam / roll number yahan likho]

---

## Overview
This project implements a complete NLP pipeline: PDF text extraction,
preprocessing, n-gram generation, Word2Vec-based next word prediction
(CBOW & Skip-gram), and text classification (Naive Bayes, Logistic
Regression, K-Means) on a 50-document corpus organized into 4 categories.

## Dataset Note
The corpus was derived from a single public-domain source — *The Will
to Believe*, William James (Project Gutenberg) — split thematically
by essay into 50 documents across 4 categories, due to time constraints
on collecting 50 independently-sourced PDFs. This is disclosed
transparently; see Limitations below.

| Category | Source Essay | Documents |
|---|---|---|
| faith_and_belief | The Will to Believe | 11 |
| life_and_pessimism | Is Life Worth Living? | 11 |
| rationality_philosophy | The Sentiment of Rationality | 16 |
| theism_and_mind | Reflex Action and Theism | 12 |

## Pipeline Steps
1. **Data Collection** — 50 PDFs organized in category folders (`data/`)
2. **Text Extraction** — `pypdf` used to extract raw text, saved to CSV
3. **Preprocessing** — lowercase, URL/email/number/punctuation removal,
   stopword removal, tokenization, short-word removal (via regex)
4. **N-gram Generation** — unigram, bigram, trigram corpora built
5. **Part A: Next Word Prediction** — CBOW and Skip-gram Word2Vec models
   trained on unigram/bigram/trigram corpora; prediction function returns
   top-N related words for an input word/phrase
6. **Part B: Text Classification** — Naive Bayes, Logistic Regression
   (on unigram/bigram/trigram features), and K-Means clustering
   (compared against true labels via ARI/NMI)
7. **Evaluation** — accuracy, precision, recall, F1-score, confusion
   matrix, PCA cluster visualization

## Results Summary
| Model | N-gram | Accuracy | F1-score |
|---|---|---|---|
| [best model apne output se fill karo] | [ngram] | [value] | [value] |

*(full results in `outputs/results_summary.txt`)*

## Key Findings
- CBOW gave higher-confidence but more surface-level predictions
  (word-form co-occurrence, e.g. "belief" → "believe"); Skip-gram gave
  lower-confidence but more thematically diverse associations —
  expected behavior on a small training corpus.
- [best model] performed best on [ngram] features for classification.
- K-Means clustering achieved ARI = [value], NMI = [value] against
  true category labels.

## How to Run
1. Open `notebook/NLP_Assignment_04.ipynb` in Google Colab
2. Upload the `data/` folder (as zip) when prompted
3. Run all cells top to bottom

## Limitations
- Single-author, single-book corpus (not 50 independently sourced
  documents) — reduces stylistic diversity between categories.
- Small dataset (50 docs, ~12 in test split) → metrics have high variance.
- Word2Vec embeddings trained on a small (~43k word) corpus are less
  robust than typically-sized training corpora.
- Bigram/trigram features are sparse on this corpus size.

## Repo Structure
```
nlp-assignment-04/
├── README.md
├── requirements.txt
├── notebook/
│   └── NLP_Assignment_04.ipynb
├── data/
│   ├── faith_and_belief/
│   ├── life_and_pessimism/
│   ├── rationality_philosophy/
│   └── theism_and_mind/
└── outputs/
    ├── nlp_dataset_final.csv
    ├── graphs/
    └── results_summary.txt
```
