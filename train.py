"""
Fake News Detection - Training Script
Author: Tilal Ahmed
Models: Logistic Regression, Decision Tree, Gradient Boosting,
        Random Forest, SVM (LinearSVC), Naive Bayes
"""

import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


# ─── 1. Load Data ────────────────────────────────────────────────────────────

def load_data(fake_path: str, true_path: str) -> pd.DataFrame:
    data_fake = pd.read_csv(fake_path)
    data_true = pd.read_csv(true_path)

    data_fake["class"] = 0
    data_true["class"] = 1

    # Remove last 10 rows reserved for manual testing
    data_fake = data_fake.iloc[:-10]
    data_true = data_true.iloc[:-10]

    data = pd.concat([data_fake, data_true], axis=0)
    data = data.drop(["title", "subject", "date"], axis=1)
    data = data.dropna()
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    return data


# ─── 2. Text Preprocessing ───────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# ─── 3. Train & Evaluate ─────────────────────────────────────────────────────

def train_and_evaluate(xv_train, xv_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree":       DecisionTreeClassifier(),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=0),
        "Random Forest":       RandomForestClassifier(random_state=0),
        "SVM (LinearSVC)":     LinearSVC(random_state=0),
        "Naive Bayes":         MultinomialNB(),
    }

    trained = {}
    for name, model in models.items():
        model.fit(xv_train, y_train)
        preds = model.predict(xv_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n{'='*50}")
        print(f"  {name}  —  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print('='*50)
        print(classification_report(y_test, preds, target_names=["Fake", "True"]))
        trained[name] = model

    return trained


# ─── 4. Save Models ──────────────────────────────────────────────────────────

def save_models(trained_models: dict, vectorizer: TfidfVectorizer):
    import os
    os.makedirs("models", exist_ok=True)

    pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
    print("\nSaved: models/vectorizer.pkl")

    name_to_file = {
        "Logistic Regression": "lr_model.pkl",
        "Decision Tree":       "dt_model.pkl",
        "Gradient Boosting":   "gb_model.pkl",
        "Random Forest":       "rf_model.pkl",
        "SVM (LinearSVC)":     "svm_model.pkl",
        "Naive Bayes":         "nb_model.pkl",
    }

    for name, model in trained_models.items():
        fname = name_to_file[name]
        pickle.dump(model, open(f"models/{fname}", "wb"))
        print(f"Saved: models/{fname}")


# ─── 5. Manual Testing ───────────────────────────────────────────────────────

def output_label(n: int) -> str:
    return "Fake News" if n == 0 else "True News"


def manual_testing(news: str, trained_models: dict, vectorizer: TfidfVectorizer):
    df = pd.DataFrame({"text": [news]})
    df["text"] = df["text"].apply(preprocess)
    xv = vectorizer.transform(df["text"])

    print("\n── Predictions ──────────────────────────────")
    for name, model in trained_models.items():
        pred = model.predict(xv)[0]
        print(f"  {name:<22}: {output_label(pred)}")
    print("─────────────────────────────────────────────")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fake News Detection Trainer")
    parser.add_argument("--fake",  default="data/Fake.csv",  help="Path to Fake.csv")
    parser.add_argument("--true",  default="data/True.csv",  help="Path to True.csv")
    parser.add_argument("--test",  default=None,             help="News text to test after training")
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.fake, args.true)
    data["text"] = data["text"].apply(preprocess)

    x = data["text"]
    y = data["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    print("Vectorizing...")
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test  = vectorizer.transform(x_test)

    print("\nTraining all models...\n")
    trained_models = train_and_evaluate(xv_train, xv_test, y_train, y_test)

    save_models(trained_models, vectorizer)

    if args.test:
        manual_testing(args.test, trained_models, vectorizer)
