import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def create_derived_features(df):
    """
    Tworzy nowe cechy na podstawie istniejących danych
    """
    # Obliczenie różnych wskaźników efektywności
    df['blueKDA'] = (df['blueKills'] + df['blueAssists']) / np.maximum(df['blueDeaths'], 1)
    df['redKDA'] = (df['redKills'] + df['redAssists']) / np.maximum(df['redDeaths'], 1)

    # Stosunek wardów do ich zniszczenia
    df['blueWardEfficiency'] = df['blueWardsDestroyed'] / np.maximum(df['redWardsPlaced'], 1)
    df['redWardEfficiency'] = df['redWardsDestroyed'] / np.maximum(df['blueWardsPlaced'], 1)

    # Obliczenie całkowitej kontroli obiektów
    df['blueObjectiveControl'] = df['blueEliteMonsters'] + df['blueTowersDestroyed']
    df['redObjectiveControl'] = df['redEliteMonsters'] + df['redTowersDestroyed']

    # Efektywność farmienia
    df['blueCSEfficiency'] = df['blueTotalMinionsKilled'] / (df['blueTotalMinionsKilled'] + df['redTotalMinionsKilled'])
    df['redCSEfficiency'] = df['redTotalMinionsKilled'] / (df['blueTotalMinionsKilled'] + df['redTotalMinionsKilled'])

    return df


def normalize_features(df):
    """
    Normalizuje wybrane cechy numeryczne
    """
    numeric_columns = [col for col in df.columns if any(x in col for x in ['Gold', 'Experience', 'CS', 'Level'])]

    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


def analyze_feature_importance(df):
    """
    Analizuje ważność cech dla wyniku gry
    """
    correlations = df.corr()['blueWins'].sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    correlations[1:16].plot(kind='bar')
    plt.title('Top 15 cech skorelowanych z wygraną niebieskiej drużyny')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')

    return correlations


def generate_preprocessing_report(df, stats, correlations):
    """
    Generuje raport z preprocessingu
    """
    report = """
    Raport z preprocessingu danych League of Legends
    ==============================================

    1. Informacje o zbiorze danych:
    ------------------------------
    - Liczba obserwacji: {}
    - Liczba cech: {}

    2. Nowe utworzone cechy:
    ----------------------
    - KDA (Kill/Death/Assist ratio)
    - Efektywność wardów
    - Kontrola obiektów
    - Efektywność farmienia

    3. Top 5 cech najsilniej skorelowanych z wygraną:
    ---------------------------------------------
    {}

    4. Podstawowe statystyki po normalizacji:
    --------------------------------------
    {}
    """.format(
        df.shape[0],
        df.shape[1],
        correlations.head().to_string(),
        stats.to_string()
    )

    return report