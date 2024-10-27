import pandas as pd
import os
from pathlib import Path
import sys

# Dodanie głównego katalogu projektu do PYTHONPATH
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

from src.utils.preprocessing_utils import (
    create_derived_features,
    normalize_features,
    analyze_feature_importance,
    generate_preprocessing_report
)


def load_and_validate_data(file_path):
    """
    Wczytuje dane i wykonuje podstawową walidację
    """
    df = pd.read_csv(file_path)

    missing_values = df.isnull().sum()
    duplicates = df.duplicated().sum()

    print(f"Kształt danych: {df.shape}")
    print(f"\nBrakujące wartości:\n{missing_values[missing_values > 0]}")
    print(f"\nLiczba duplikatów: {duplicates}")

    return df

def preprocess_data():
    """
    Główna funkcja przetwarzająca dane
    """
    # Ustawienie ścieżek
    project_dir = Path(__file__).resolve().parents[2]
    input_file = project_dir / "data/raw/high_diamond_ranked_10min.csv"
    output_file = project_dir / "data/processed/processed_league_data.csv"
    report_file = project_dir / "reports/preprocessing_report.txt"

    # Utworzenie katalogów jeśli nie istnieją
    os.makedirs(project_dir / "data/processed", exist_ok=True)
    os.makedirs(project_dir / "reports/figures", exist_ok=True)

    print("1. Wczytywanie i walidacja danych...")
    df = load_and_validate_data(input_file)

    print("\n2. Tworzenie nowych cech...")
    df = create_derived_features(df)

    print("\n3. Normalizacja wybranych cech...")
    df = normalize_features(df)

    print("\n4. Analiza ważności cech...")
    correlations = analyze_feature_importance(df)

    # Podstawowe statystyki
    stats = df.describe()

    # Generowanie i zapis raportu
    report = generate_preprocessing_report(df, stats, correlations)
    with open(report_file, "w") as f:
        f.write(report)

    # Zapis przetworzonego zbioru danych
    df.to_csv(output_file, index=False)

    print(f"\nPrzetwarzanie zakończone. Wyniki zapisane w {output_file}")
    print(f"Raport zapisany w {report_file}")


if __name__ == "__main__":
    preprocess_data()