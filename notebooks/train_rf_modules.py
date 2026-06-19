"""
Обучение Random Forest на модульных признаках
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

# загрузка данных
df = pd.read_csv("module_features_dataset.csv")
print(f"Загружено геномов: {len(df)}")

# чтение списка признаков (без probiotic_potential)
with open("module_feature_cols.txt", "r") as f:
    feature_cols = [line.strip() for line in f]

target_col = "label"

X = df[feature_cols]
y = df[target_col].astype(int)

print(f"\nЧисло признаков: {len(feature_cols)}")
print(f"Распределение классов:")
print(y.value_counts())

# 1. Разделение на train/test
# Исключаем MAG из обучения
mags = ['bin.1', 'bin.2', 'bin.3', 'bin.4']
df_mag = df[df['genome'].isin(mags)].copy()
df_train = df[~df['genome'].isin(mags)].copy()

print(f"\nДанные для обучения: {len(df_train)} (MAG удалены)")
print(f"MAG для тестирования: {len(df_mag)}")

X_train = df_train[feature_cols]
y_train = df_train[target_col].astype(int)

# Обычное разделение на train/val
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 2. Балансировка классов (smote)
print(f"  До SMOTE: класс 0={sum(y_train_split==0)}, класс 1={sum(y_train_split==1)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_split, y_train_split)

print(f"  После SMOTE: класс 0={sum(y_train_balanced==0)}, класс 1={sum(y_train_balanced==1)}")


# 3. Обучение модели
print(f"\nОбучение Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_balanced, y_train_balanced)

# 4. Оценка на валидации
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

print("Результаты на валидационной выборке")
print(classification_report(y_val, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_val, y_proba):.3f}")
print(f"Cross-validation F1: {cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='f1').mean():.3f}")

# 5. Оценка на MAG
if len(df_mag) > 0:
    X_mag = df_mag[feature_cols]
    y_mag_true = df_mag[target_col].astype(int)
    y_mag_pred = model.predict(X_mag)
    y_mag_proba = model.predict_proba(X_mag)[:, 1]
    
    print("Результаты на MAG")
    for i, (idx, row) in enumerate(df_mag.iterrows()):
        print(f"{row['genome']}: true={int(y_mag_true.iloc[i])}, pred={y_mag_pred[i]}, proba={y_mag_proba[i]:.3f}")


# 6. Важность признаков
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

print("Важность признаков")
print(feat_imp)

# Визуализация
plt.figure(figsize=(10, 8))
feat_imp.plot(kind='bar')
plt.title('Feature Importance (Module-based Features)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance_modules.png', dpi=150)
plt.show()

# 7. Дополнительный анализ: сравнение с probiotic_potential
if 'probiotic_potential' in df.columns:
    print("Дополнительный анализ: probiotic_potential")
    
    # Корреляция между предсказаниями модели и probiotic_potential
    # Для MAG
    mag_proba = y_mag_proba
    mag_potential = df_mag['probiotic_potential'].values
    
    print("\nMAG: предсказанная вероятность vs probiotic_potential")
    for i, (idx, row) in enumerate(df_mag.iterrows()):
        print(f"{row['genome']}: proba={mag_proba[i]:.3f}, potential={mag_potential[i]:.3f}")
    
    # Корреляция на всей выборке
    all_proba = model.predict_proba(df[feature_cols])[:, 1]
    all_potential = df['probiotic_potential'].values
    corr = np.corrcoef(all_proba, all_potential)[0, 1]
    print(f"\nКорреляция (предсказания vs probiotic_potential): {corr:.3f}")


# 8. Матрица ошибок и ROC-AUC
# Матрица ошибок
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Validation)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_modules.png', dpi=150)
plt.show()

# ROC-AUC
fpr, tpr, _ = roc_curve(y_val, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_val, y_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve_modules.png', dpi=150)
plt.show()


# 9. Сохранение модели
joblib.dump(model, "random_forest_modules.pkl")
print("\n✅ Модель сохранена в 'random_forest_modules.pkl'")