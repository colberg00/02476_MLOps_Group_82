Baseline model: TF-IDF + Logistic Regression

This script trains a baseline text classifier to predict the news outlet (Fox News vs NBC News) from URL slug text.

The slug text is derived from the article URL and serves as a proxy for the article headline.
Domain names, section labels, and outlet identifiers are removed during preprocessing to avoid label leakage.

To make this work you must:

1. Download the dataset

Download the dataset from Hugging Face:
https://huggingface.co/datasets/Jia555/cis519_news_urls
Place the raw file(s) inside the following directory:
data/cis519_news_urls/


2. Preprocess the data

From the project root, run:
uv run python -m mlops_course_project.data
This will:
parse the raw URLs
extract and clean the URL slug text
assign outlet labels (fox / nbc)
create stratified train/validation/test splits
The processed files are written to:
data/processed/
├── train.csv
├── val.csv
└── test.csv

Each file contains two columns:
slug: cleaned URL slug text (headline proxy)
outlet: news outlet label (fox or nbc)

3. Train the baseline model

Run the training script from the project root:
uv run python -m mlops_course_project.train
This trains a TF-IDF + Logistic Regression model and evaluates it on the validation and test sets.

Outputs:
trained model: models/baseline.joblib
metrics report: reports/baseline_metrics.json

4. Optional hyperparameters

You may override the default hyperparameters:
uv run python -m mlops_course_project.train \
  --ngram-max 3 \
  --max-features 100000 \
  --c 2.0

  ngram-max default is 2
  max-features default is 50000
  c default is 1.0

  