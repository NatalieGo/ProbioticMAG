import pandas as pd
import os
import re
from pathlib import Path

def extract_probiotic_features(tsv_file):
    """Извлекает признаки пробиотического потенциала из файла Bakta"""
    try:
        # Имена колонок для Bakta
        column_names = [
            'sequence_id', 'type', 'start', 'stop', 'strand',
            'locus_tag', 'gene', 'product', 'db_xrefs'
        ]
        
        # Чтение файла
        df = pd.read_csv(tsv_file, sep='\t', comment='#', header=None, names=column_names, encoding='utf-8')
        
        # Проверка колонки product
        if 'product' not in df.columns:
            print(f"Колонка 'product' не найдена в {tsv_file.name}")
            return None
        
        features = {'genome': Path(tsv_file).parent.name}
        
        # Положительные маркеры (пробиотические признаки)
        positive_patterns = {
            'sortase': r'sortase',
            'lpxtg': r'LPXTG',
            'collagen_binding': r'collagen',
            'dnaK': r'dnaK',
            'groEL': r'groEL',
            'clp': r'clp[PX]',
            'bacteriocin': r'bacteriocin|lantibiotic|enterocin|nisin|pediocin|plantaricin',
            'crispr_cas': r'crispr|cas9|cas1|cas2|cas7|cms',
            'adhesion': r'adhesion|adhesin|mucus|mucin',
            'bsh': r'bile|bsh|choloylglycine',
            'stress_universal': r'universal stress',
            'methyltransferase': r'methyltransferase|dna-methyltransferase',
            'clp_protease': r'clp protease|ATP-dependent clp',
            'phage_resistance': r'phage resistance|abortive infection|abi',
        }
        
        # Отрицательные маркеры (факторы риска)
        negative_patterns = {
            'virulence': r'virulence|toxin|exotoxin|hemolysin|enterotoxin|staphylococcal',
            'amr': r'multidrug resistance|antibiotic resistance|quinolone resistance|mec|van|tet',
            'phage': r'phage|prophage|integrase|transposase',
        }
        
        for name, pattern in positive_patterns.items():
            count = len(df[df['product'].str.contains(pattern, case=False, na=False)])
            features[f'{name}_count'] = count
            features[f'{name}_present'] = 1 if count > 0 else 0
        
        for name, pattern in negative_patterns.items():
            count = len(df[df['product'].str.contains(pattern, case=False, na=False)])
            features[f'{name}_count'] = count
            features[f'{name}_present'] = 1 if count > 0 else 0
        
        # Общие метрики
        features['total_genes'] = len(df)
        features['hypothetical_count'] = len(df[df['product'].str.contains('hypothetical', case=False, na=False)])
        
        return features
    
    except Exception as e:
        print(f"Ошибка при обработке {tsv_file}: {e}")
        return None

# Обработка результатов
results_dir = Path("./bakta_results")
all_features = []

print("Обработка файлов...")

# Отладка
for tsv_file in results_dir.glob("*/*.tsv"):
    if 'hypotheticals' in str(tsv_file) or 'inference' in str(tsv_file):
        continue
    print(f"\nПример содержимого {tsv_file.name}:")
    with open(tsv_file, 'r') as f:
        lines = f.readlines()[:3]
        for line in lines:
            if not line.startswith('#'):
                print(f"  {line.strip()}")
    break

print("Старт обработки...")

# Обработка всех файлов
count = 0
for tsv_file in results_dir.glob("*/*.tsv"):
    # Вспомогательные файлы пропускаем
    if 'hypotheticals' in str(tsv_file) or 'inference' in str(tsv_file):
        continue
    
    features = extract_probiotic_features(tsv_file)
    if features:
        all_features.append(features)
        count += 1
        if count % 50 == 0:
            print(f"  ... обработано {count} геномов")

# Сохранение результатов
if all_features:
    df = pd.DataFrame(all_features)
    df.to_csv("genome_features.csv", index=False)
    print("\n" + "=" * 60)
    print(f"Всего обработано геномов: {len(df)}")
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nСтатистика по признакам:")
    print(df.describe())
    print("\nРаспределение пробиотических маркеров:")
    marker_cols = [col for col in df.columns if '_present' in col]
    for col in marker_cols:
        print(f"  {col}: {df[col].sum()} / {len(df)}")
else:
    print("\nНет данных для сохранения.")