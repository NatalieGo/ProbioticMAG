"""
SHAP-анализ для интерпретации предсказаний XGBoost
Показывает, какие признаки влияют на решение и как
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

print("SHAP-анализ модели XGBoost")

df = pd.read_csv("module_features_dataset.csv")
print(f"Загружено геномов: {len(df)}")

with open("module_feature_cols.txt", "r") as f:
    feature_cols = [line.strip() for line in f]

target_col = "label"

mags = ['bin.1', 'bin.2', 'bin.3', 'bin.4']
df_mag = df[df['genome'].isin(mags)].copy()
df_train = df[~df['genome'].isin(mags)].copy()

X = df_train[feature_cols]
y = df_train[target_col].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

xgb_model = joblib.load("xgboost_modules.pkl")
print("Модель XGBoost загружена")

print("Создание SHAP explainer")
explainer = shap.TreeExplainer(xgb_model)
print("SHAP explainer готов")

shap_values_val = explainer.shap_values(X_val)

X_mag = df_mag[feature_cols]
shap_values_mag = explainer.shap_values(X_mag)

print("Построение summary plot")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_val, X_val, feature_names=feature_cols, show=False)
plt.title('SHAP Summary Plot: Feature Importance with Direction', fontsize=14)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Сохранено: shap_summary_plot.png")

print("Построение bar plot")

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_val, X_val, feature_names=feature_cols, 
                  plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Global)', fontsize=14)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("Сохранено: shap_bar_plot.png")

print("Детальный анализ MAG")

os.makedirs("shap_waterfalls", exist_ok=True)

for i, (idx, row) in enumerate(df_mag.iterrows()):
    mag_name = row['genome']
    true_label = int(row['label'])
    pred_proba = xgb_model.predict_proba(X_mag.iloc[i:i+1])[0, 1]
    
    print(f"\nMAG: {mag_name}")
    print(f"Истинная метка: {true_label} (1=пробиотик, 0=не-пробиотик)")
    print(f"Предсказанная вероятность: {pred_proba:.3f}")
    
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_mag[i],
            base_values=explainer.expected_value,
            data=X_mag.iloc[i],
            feature_names=feature_cols
        ),
        show=False,
        max_display=10
    )
    plt.title(f'SHAP Waterfall Plot: {mag_name} (pred={pred_proba:.3f}, true={true_label})')
    plt.tight_layout()
    plt.savefig(f'shap_waterfalls/{mag_name}_waterfall.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Ключевые факторы для {mag_name}:")
    shap_df = pd.DataFrame({
        'feature': feature_cols,
        'shap_value': shap_values_mag[i],
        'feature_value': X_mag.iloc[i].values
    })
    shap_df = shap_df.sort_values('shap_value', ascending=False)
    
    print("Факторы, повышающие вероятность пробиотика:")
    positive = shap_df[shap_df['shap_value'] > 0].head(3)
    if len(positive) > 0:
        for _, row_f in positive.iterrows():
            print(f"  + {row_f['feature']}: {row_f['shap_value']:.3f} (значение={row_f['feature_value']:.3f})")
    else:
        print("  (нет)")
    
    print("Факторы, понижающие вероятность пробиотика:")
    negative = shap_df[shap_df['shap_value'] < 0].head(3)
    if len(negative) > 0:
        for _, row_f in negative.iterrows():
            print(f"  - {row_f['feature']}: {row_f['shap_value']:.3f} (значение={row_f['feature_value']:.3f})")
    else:
        print("  (нет)")

print("Waterfall plots сохранены в папку shap_waterfalls/")

print("Сравнительный анализ: bin.2 vs bin.4")

idx_bin2 = df_mag[df_mag['genome'] == 'bin.2'].index[0] - df_mag.index[0]
idx_bin4 = df_mag[df_mag['genome'] == 'bin.4'].index[0] - df_mag.index[0]

print(f"\nbin.2 (Streptococcus thermophilus) - пробиотик:")
print(f"  Предсказание: {xgb_model.predict_proba(X_mag.iloc[idx_bin2:idx_bin2+1])[0, 1]:.3f}")
print(f"  SHAP значения:")
shap_bin2 = pd.Series(shap_values_mag[idx_bin2], index=feature_cols).sort_values(ascending=False)
print(shap_bin2.head(5).round(3))

print(f"\nbin.4 (Leuconostoc mesenteroides) - ошибка модели:")
print(f"  Предсказание: {xgb_model.predict_proba(X_mag.iloc[idx_bin4:idx_bin4+1])[0, 1]:.3f}")
print(f"  SHAP значения:")
shap_bin4 = pd.Series(shap_values_mag[idx_bin4], index=feature_cols).sort_values(ascending=False)
print(shap_bin4.head(5).round(3))

print("Создание force plots для MAG...")

for i, (idx, row) in enumerate(df_mag.iterrows()):
    mag_name = row['genome']
    true_label = int(row['label'])
    pred_proba = xgb_model.predict_proba(X_mag.iloc[i:i+1])[0, 1]
    
    plt.figure(figsize=(14, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values_mag[i],
        X_mag.iloc[i],
        feature_names=feature_cols,
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot: {mag_name} (pred={pred_proba:.3f}, true={true_label})')
    plt.tight_layout()
    plt.savefig(f'shap_waterfalls/{mag_name}_force.png', dpi=150, bbox_inches='tight')
    plt.show()

print("Force plots сохранены")

print("Сводная таблица SHAP для MAG:")

shap_mag_df = pd.DataFrame(shap_values_mag, columns=feature_cols, index=df_mag['genome'])
shap_mag_df['pred_proba'] = xgb_model.predict_proba(X_mag)[:, 1]
shap_mag_df['true_label'] = df_mag['label'].values

print(shap_mag_df.round(3))
shap_mag_df.to_csv("shap_mag_analysis.csv")
print("Сохранено: shap_mag_analysis.csv")