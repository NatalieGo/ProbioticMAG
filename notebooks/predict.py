# predict.py
"""Предсказание на новых геномах с использованием обученных моделей"""

import pandas as pd
import joblib
import sys

def predict_new_genomes(input_file, output_file="predictions.csv"):
    df = pd.read_csv(input_file)
    # ... логика построения признаков ...
    rf_model = joblib.load("random_forest_modules.pkl")
    xgb_model = joblib.load("xgboost_modules.pkl")
    # ... предсказания ...
    df.to_csv(output_file)

if __name__ == "__main__":
    predict_new_genomes(sys.argv[1])