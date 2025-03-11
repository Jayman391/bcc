# Behavior & Content Comparisons

## Overview
This project aims to compare and evaluate text classification models and LLM inference models on behavioral and content data. It involves preprocessing data, extracting embeddings, sentiment analysis, modeling via ensemble techniques, inference through ollama models, and comprehensive analytics.


## Preprocessing

**Textacy Module:**  
Utilize the [Textacy Python library](https://www.geeksforgeeks.org/textacy-module-in-python/) for preprocessing text data. Textacy helps with text cleaning, tokenization, lemmatization, normalization, and linguistic feature extraction.

---

## Features

**Sentence-Transformer Text Embeddings:**  
Embedding features will be generated using various pretrained [SentenceTransformer models](https://huggingface.co/models?library=sentence-transformers&sort=downloads).

**Sentiment Analysis:**  
Apply sentiment analysis using [VADER Sentiment Analysis](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/), a lexicon and rule-based sentiment analyzer designed explicitly for social media texts.

---

## Modeling

### Model Comparison
Use a variety of classification algorithms available in [scikit-learn](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) for comparison and evaluation.

### Model Architecture
- **Ensemble of SentenceTransformer models**
- **Ensemble of all sklearn estimators**

### Sklearn Model Selection Criterion
- Define a custom likelihood function:
  
  $$\text{Likelihood} = \text{Average}(AUC_{ROC_{One V All}}) \times F1_{Micro}$$

- Select the best-performing model based on this metric.

---

## Inference

### Ensemble of Ollama Models
- Use an ensemble of various Ollama inference models for predicting probabilities and labels on remaining unlabeled data after initial classification.

---

## Analytics

### SentenceTransformer Models Analytics
- **AUC ROC plots**
- **Classification Tables**
- **Correlation Matrices**
- **Topic Modeling** of predicted labels

### Ollama Models Analytics
- **Correlation Matrices**
- **Topic Modeling** of predicted labels

### Cross-Model Comparisons (SentenceTransformer vs. Ollama)
- **Correlation Matrix** (SentenceTransformer models along one dimension, Ollama models on the other dimension)
- **Topic Modeling** of predicted labels across both sets of models

---

## Deployment on VACC
The pipeline is designed to run efficiently on the Vermont Advanced Computing Core (VACC):

- **CPU Scripts**: Classification ensemble & analytics tasks.
- **GPU Scripts**: Ollama inference tasks.

SLURM scripts provided ensure seamless and optimized HPC usage.

---
