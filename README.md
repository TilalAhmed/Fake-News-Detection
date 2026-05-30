[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fake-news-detection-t2kfj9uxbdhtzabcpjx4az.streamlit.app/)
# Fake News Detection System

A machine learning project that classifies news articles as **Fake** or **True** using six different classification algorithms trained on TF-IDF text features.

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~98.6% |
| Decision Tree | ~99.5% |
| Gradient Boosting | ~99.5% |
| Random Forest | ~98.9% |
| SVM (LinearSVC) | ~99.4% |
| Naive Bayes | ~93.4% |

---
> **Note:** Model trained on ISOT dataset (2016-2017 political news). Performance may vary on modern news articles. Random Forest excluded from live app due to file size.
## Dataset

[ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-and-real-news-dataset) — ~45,000 news articles (Fake.csv + True.csv).

Place the downloaded CSV files inside a `data/` folder:
```
fake-news-detection/
└── data/
    ├── Fake.csv
    └── True.csv
```

---

## Project Structure

```
fake-news-detection/
├── src/
│   └── train.py          # Training script — all 6 models
├── notebooks/
│   └── Fake_News_Detection_System_Tilal_Ahmed.ipynb
├── data/                 # Add Fake.csv and True.csv here (not tracked by git)
├── models/               # Saved .pkl files appear here after training
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Clone the repo and install dependencies**
```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

**2. Add the dataset**

Download the ISOT dataset from Kaggle and place `Fake.csv` and `True.csv` inside the `data/` folder.

**3. Train all models**
```bash
python src/train.py
```

**4. Test with custom news text**
```bash
python src/train.py --test "Your news article text here"
```

Saved model files will appear in the `models/` folder as `.pkl` files.

---

## How It Works

1. **Preprocessing** — Lowercasing, removing URLs, punctuation, HTML tags, and numbers
2. **Vectorization** — TF-IDF (Term Frequency–Inverse Document Frequency) converts text to numerical features
3. **Training** — Six classifiers are trained and evaluated on a 75/25 train-test split
4. **Prediction** — All six models vote on whether a given article is Fake or True

---

## Tech Stack

- Python 3.x
- Scikit-learn (TF-IDF, all classifiers)
- Pandas, NumPy
- Matplotlib, Seaborn

---

## Author

**Tilal Ahmed**  
BS Computer Science — Iqra University, Karachi  
[LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN) · tilalahmed956@gmail.com
