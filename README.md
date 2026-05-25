# Fake News Detection 🗞️

A machine learning project that classifies news articles as real or fake using NLP techniques.

---

## Overview

Trained and compared 6 classification models on a labeled news dataset using TF-IDF 
vectorization. The best performing model achieved 99.5% accuracy.

---

## Models Used

- Support Vector Machine (SVM)
- Naïve Bayes
- Decision Tree
- Logistic Regression
- Random Forest
- Passive Aggressive Classifier

---

## Results

| Model | Accuracy |
|-------|----------|
| SVM | 99.5% |
| Logistic Regression | 99.2% |
| Random Forest | 98.9% |
| Passive Aggressive | 98.7% |
| Decision Tree | 97.8% |
| Naïve Bayes | 94.1% |

---

## Tech Stack

- **Language:** Python
- **Libraries:** Scikit-learn · Pandas · NumPy · Matplotlib
- **Technique:** TF-IDF Vectorization

---

## Dataset

Uses a publicly available fake news dataset.  
Download `True.csv` and `Fake.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them in the project root before running.
