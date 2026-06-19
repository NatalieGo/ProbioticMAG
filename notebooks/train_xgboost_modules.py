"""
Обучение XGBoost на модульных признаках и сравнение с Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib


# 1. Загрузка данных
df = pd.read_csv("module_features_dataset.csv")
print(f"Загружено геномов: {len(df)}")

# Чтение списка признаков
with open("module_feature_cols.txt", "r") as f:
    feature_cols = [line.strip() for line in f]

target_col = "label"

X = df[feature_cols]
y = df[target_col].astype(int)

print(f"\nКоличество признаков: {len(feature_cols)}")
print(f"Распределение классов:")
print(y.value_counts())


# 2. Разделение на train/test
mags = ['bin.1', 'bin.2', 'bin.3', 'bin.4']
df_mag = df[df['genome'].isin(mags)].copy()
df_train = df[~df['genome'].isin(mags)].copy()

print(f"\nДанные для обучения: {len(df_train)} (MAG удалены)")
print(f"MAG для тестирования: {len(df_mag)}")

X_train = df_train[feature_cols]
y_train = df_train[target_col].astype(int)

# Разделение на train/val
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)


# 3. Балансировка классов (smote) для обеих моделей
print(f"\nБалансировка классов через SMOTE...")
print(f"  До SMOTE: класс 0={sum(y_train_split==0)}, класс 1={sum(y_train_split==1)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)

print(f"  После SMOTE: класс 0={sum(y_train_balanced==0)}, класс 1={sum(y_train_balanced==1)}")


# 4. Обучение Random Forest
print("Обучение Random Forest")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_balanced, y_train_balanced)

# Предсказания RF
y_pred_rf = rf_model.predict(X_val)
y_proba_rf = rf_model.predict_proba(X_val)[:, 1]

print("\nРезультаты Random Forest:")
print(classification_report(y_val, y_pred_rf))
print(f"AUC-ROC: {roc_auc_score(y_val, y_proba_rf):.3f}")

# RF на MAG
X_mag = df_mag[feature_cols]
y_mag_true = df_mag[target_col].astype(int)
y_proba_rf_mag = rf_model.predict_proba(X_mag)[:, 1]


# 5. Обучение XGBoost
print("Обучение XGBoost")

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(X_train_balanced, y_train_balanced)

# Предсказания XGBoost
y_pred_xgb = xgb_model.predict(X_val)
y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

print("\nРезультаты XGBoost")
print(classification_report(y_val, y_pred_xgb))
print(f"AUC-ROC: {roc_auc_score(y_val, y_proba_xgb):.3f}")

# XGBoost на MAG
y_proba_xgb_mag = xgb_model.predict_proba(X_mag)[:, 1]


# 6. Кросс-валидация для XGBoost
print("Кросс-валидация XGBoost (5-fold)")

cv_scores = cross_val_score(
    xgb_model, X_train_balanced, y_train_balanced,
    cv=5, scoring="roc_auc"
)

print(f"CV AUC scores: {cv_scores}")
print(f"Mean CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


# 7. Сравнение моделей
print("Сравнение моделей")

# Метрики для сравнения
rf_auc = roc_auc_score(y_val, y_proba_rf)
xgb_auc = roc_auc_score(y_val, y_proba_xgb)

from sklearn.metrics import accuracy_score
rf_acc = accuracy_score(y_val, y_pred_rf)
xgb_acc = accuracy_score(y_val, y_pred_xgb)

# F1 для класса 1 (пробиотики)
rf_f1_1 = classification_report(y_val, y_pred_rf, output_dict=True)['1']['f1-score']
xgb_f1_1 = classification_report(y_val, y_pred_xgb, output_dict=True)['1']['f1-score']

# Precision/Recall для класса 1
rf_prec_1 = classification_report(y_val, y_pred_rf, output_dict=True)['1']['precision']
xgb_prec_1 = classification_report(y_val, y_pred_xgb, output_dict=True)['1']['precision']
rf_rec_1 = classification_report(y_val, y_pred_rf, output_dict=True)['1']['recall']
xgb_rec_1 = classification_report(y_val, y_pred_xgb, output_dict=True)['1']['recall']

results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "AUC-ROC": [rf_auc, xgb_auc],
    "Accuracy": [rf_acc, xgb_acc],
    "F1 (пробиотики)": [rf_f1_1, xgb_f1_1],
    "Precision (пробиотики)": [rf_prec_1, xgb_prec_1],
    "Recall (пробиотики)": [rf_rec_1, xgb_rec_1]
})

print(results.round(3))


# 8. Сравнение на MAG
print("\n" + "=" * 60)
print("ПРЕДСКАЗАНИЯ НА MAG")
print("=" * 60)

mag_comparison = pd.DataFrame({
    'MAG': df_mag['genome'].values,
    'True': y_mag_true.values,
    'RF_proba': y_proba_rf_mag,
    'XGB_proba': y_proba_xgb_mag
})
print(mag_comparison.round(3))


# 9. Важность призноков (XGBoost)
importances_xgb = xgb_model.feature_importances_
feat_imp_xgb = pd.Series(importances_xgb, index=feature_cols).sort_values(ascending=False)

print("Важность призноков (XGBoost)")
print(feat_imp_xgb)


# 10. Сравнение важности признаков (RF vs XGBoost)
importances_rf = rf_model.feature_importances_
feat_imp_rf = pd.Series(importances_rf, index=feature_cols).sort_values(ascending=False)

importance_comparison = pd.DataFrame({
    'Random Forest': feat_imp_rf,
    'XGBoost': feat_imp_xgb
})
print("\n" + "=" * 60)
print("Сравнение важности признаков")
print("=" * 60)
print(importance_comparison.round(3))

# 11. Визуализация
# ROC-кривые сравнения
plt.figure(figsize=(8, 6))

fpr_rf, tpr_rf, _ = roc_curve(y_val, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, y_proba_xgb)

plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})', linewidth=2)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Random Forest vs XGBoost')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('roc_curves_comparison.png', dpi=150)
plt.show()

# Сравнение важности признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

feat_imp_rf.head(10).plot(kind='bar', ax=axes[0])
axes[0].set_title('Random Forest Feature Importance')
axes[0].set_xlabel('Feature')
axes[0].set_ylabel('Importance')
axes[0].tick_params(axis='x', rotation=45)

feat_imp_xgb.head(10).plot(kind='bar', ax=axes[1])
axes[1].set_title('XGBoost Feature Importance')
axes[1].set_xlabel('Feature')
axes[1].set_ylabel('Importance')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150)
plt.show()


# 12. Сохранение моделей
joblib.dump(rf_model, "random_forest_modules.pkl")
joblib.dump(xgb_model, "xgboost_modules.pkl")
print("\nМодели сохранены:")
print("   - random_forest_modules.pkl")
print("   - xgboost_modules.pkl")

# Результаты сравнения
results.to_csv("model_comparison.csv", index=False)
print("\nРезультаты сравнения сохранены в 'model_comparison.csv'")