"""
Создание модульных признаков для оценки пробиотического потенциала
Из исходных признаков (counts) строятся функциональные модули
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_and_prepare_data():
    """Загрузка исходных данных"""
    df = pd.read_csv("all_genome_features_with_mags.csv")
    print(f"Загружено геномов: {len(df)}")
    
    df_labels = pd.read_csv("labels_1.csv", sep=';')
    
    mag_labels = pd.DataFrame([
        {'accession': 'bin.1', 'label': 0},
        {'accession': 'bin.2', 'label': 1},
        {'accession': 'bin.3', 'label': 1},
        {'accession': 'bin.4', 'label': 1},
    ])
    df_all_labels = pd.concat([df_labels[['accession', 'label']], mag_labels], ignore_index=True)
    df_all_labels = df_all_labels.drop_duplicates(subset=['accession'], keep='first')
    
    def extract_accession(name):
        match = re.search(r'(GCF_\d+\.\d+)', str(name))
        return match.group(1) if match else name
    
    df['accession'] = df['genome'].apply(extract_accession)
    
    df = df.merge(df_all_labels, on='accession', how='left')
    
    df = df[df['label'].notna()]
    df['label'] = df['label'].astype(int)
    
    print(f"После объединения с метками: {len(df)} геномов")
    print(f"  Пробиотиков (1): {(df['label'] == 1).sum()}")
    print(f"  Не-пробиотиков (0): {(df['label'] == 0).sum()}")
    
    return df

def build_module_features(df):
    """
    Построение модульных признаков из count-признаков
    
    Модули:
    - Adhesion: способность к адгезии и колонизации
    - Antimicrobial: антимикробная активность (бактериоцины)
    - Phage defense: защита от бактериофагов
    - Stress response: устойчивость к стрессам
    - Risk: факторы риска (AMR, вирулентность, мобильные элементы)
    """
    df = df.copy()
    
    df["adhesion_module"] = df.get("sortase_count", 0) + df.get("lpxtg_count", 0)
    
    df["antimicrobial_module"] = df.get("bacteriocin_count", 0)
    
    df["phage_defense_module"] = df.get("crispr_cas_count", 0) + df.get("phage_resistance_count", 0)
    
    df["stress_module"] = df.get("stress_universal_count", 0) + df.get("clp_protease_count", 0)
    
    df["amr_module"] = df.get("amr_count", 0)
    df["virulence_module"] = df.get("virulence_count", 0)
    df["mobile_elements_module"] = df.get("phage_count", 0)
    
    total_genes = df["total_genes"].values
    df["adhesion_norm"] = df["adhesion_module"] / (total_genes + 1)
    df["stress_norm"] = df["stress_module"] / (total_genes + 1)
    df["phage_defense_norm"] = df["phage_defense_module"] / (total_genes + 1)
    df["amr_norm"] = df["amr_module"] / (total_genes + 1)
    df["virulence_norm"] = df["virulence_module"] / (total_genes + 1)
    
    df["defense_vs_risk"] = (
        df["antimicrobial_module"] + df["phage_defense_module"]
    ) / (df["amr_module"] + df["virulence_module"] + 1)
    
    df["survival_vs_mobile"] = df["stress_module"] / (df["mobile_elements_module"] + 1)
    
    df["adhesion_vs_genome"] = df["adhesion_module"] / (total_genes + 1)
    
    df["probiotic_potential"] = (
        df["adhesion_norm"] * 2 +
        df["antimicrobial_module"] / 5 +
        df["phage_defense_norm"] * 3 +
        df["stress_norm"] * 2 -
        df["amr_norm"] * 5 -
        df["virulence_norm"] * 3
    )
    
    return df

def select_features(df):
    """Выбор признаков для модели"""
    
    feature_cols = [
        "adhesion_module",
        "antimicrobial_module",
        "phage_defense_module",
        "stress_module",
        "amr_module",
        "virulence_module",
        "mobile_elements_module",
        
        "adhesion_norm",
        "stress_norm",
        "phage_defense_norm",
        
        "defense_vs_risk",
        "survival_vs_mobile",
        "adhesion_vs_genome",
    ]
    
    return feature_cols

def main():
    print("Построение модльных признаков")
    
    df = load_and_prepare_data()
    
    df = build_module_features(df)
    
    feature_cols = select_features(df)
    target_col = "label"
    
    print(f"\nПризнаки для модели ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  - {f}")
    
    print(f"\nДополнительный признак (только для анализа):")
    print("  - probiotic_potential")
    
    df.to_csv("module_features_dataset.csv", index=False)
    print(f"\nСохранено в 'module_features_dataset.csv'")
    
    print(f"\nСтатистика модульных признаков:")
    print(df[feature_cols].describe())
    
    with open("module_feature_cols.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")
    
    with open("all_module_features.txt", "w") as f:
        all_cols = feature_cols + ["probiotic_potential"]
        for col in all_cols:
            f.write(col + "\n")
    
    return df, feature_cols

if __name__ == "__main__":
    df, feature_cols = main()