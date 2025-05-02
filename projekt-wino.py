# Importowanie bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.ioff()  # Wyłączenie trybu interakcyjnego dla matplotlib


# ---- Funkcja do wczytywania i czyszczenia danych ----
def load_and_clean_data(file_path: str, sep: str = ";") -> pd.DataFrame:
    data = pd.read_csv(file_path, sep=sep)
    
    # Konwersja kolumn do typu float, z wyjątkiem zmiennej celu
    for col in data.columns[:-1]:
        data[col] = data[col].astype(float)
    
    # Sprawdzenie i usunięcie brakujących wartości
    missing_values = data.isnull().sum()
    if missing_values.any():
        print("Brakujące wartości w kolumnach:\n", missing_values)
        data = data.dropna()
        print("Usunięto wiersze z brakującymi wartościami.\n")
    
    print("Podstawowe informacje o danych (po czyszczeniu):")
    print(data.info())
    return data


# ---- Funkcja do eksploracyjnej analizy danych ----
def visualize_data_distribution(data: pd.DataFrame):
    # Histogramy
    data.hist(bins=15, layout=(4, 3), figsize=(15, 10))
    plt.suptitle("Histogramy zmiennych", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Rozkłady zmiennych
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns):
        plt.subplot(4, 3, i + 1)
        sns.kdeplot(data[column], fill=True)
        plt.xlabel(column)
        plt.ylabel("Gęstość")
    plt.suptitle("Rozkład zmiennych", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Macierz korelacji
    plt.figure(figsize=(12, 9))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Macierz korelacji", fontsize=16)
    plt.show()


# ---- Funkcja do macierzy pomyłek ----
def plot_confusion_matrix(cm, classes, title):
   
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title(title, fontsize=16)
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Rzeczywista klasa")
    plt.tight_layout()
    plt.show()


# ---- Funkcja do trenowania klasyfikatorów ----
def train_classifiers(X_train, X_test, y_train, y_test):
   
    # Drzewo Decyzyjne
    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

     # Wizualizacja drzewa decyzyjnego
    plt.figure(figsize=(15, 10))
    plot_tree(tree_model, feature_names=X_train.columns, class_names=[str(i) for i in np.unique(y_train)],
              filled=True, fontsize=8, rounded=True)
    plt.title("Wizualizacja drzewa decyzyjnego", fontsize=16)
    plt.show()

    # Raport klasyfikacji dla Drzewa Decyzyjnego
    print("\nDrzewo decyzyjne - raport klasyfikacji:\n", classification_report(y_test, y_pred_tree, zero_division=0))
    cm_tree = confusion_matrix(y_test, y_pred_tree)
    plot_confusion_matrix(cm_tree, np.unique(y_test), "Macierz pomyłek - Drzewo decyzyjne")
  
    # Skalowanie danych dla k-NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # k-Nearest Neighbors (k-NN)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)

    # Raport klasyfikacji dla k-NN
    print("\nk-NN - raport klasyfikacji:\n", classification_report(y_test, y_pred_knn, zero_division=0))
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    plot_confusion_matrix(cm_knn, np.unique(y_test), "Macierz pomyłek - k-NN")

    return tree_model, knn_model


# ---- Funkcja do wykresu krzywej uczenia ----
def plot_learning_curves(models, X_train, y_train):
    plt.figure(figsize=(10, 6))
    for model_name, model in models.items():
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
        )
        plt.plot(train_sizes, np.mean(test_scores, axis=1), label=model_name)
    plt.title("Krzywe uczenia", fontsize=16)
    plt.xlabel("Rozmiar zbioru treningowego")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid()
    plt.show()


# ---- Funkcja do regresji ----
def train_regressors(X_train, X_test, y_train, y_test):
    # Regresja liniowa
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    y_pred_lin_rounded = np.round(y_pred_lin)

    print("\nRegresja liniowa - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin_rounded)))

    # Random Forest
    rf_reg = RandomForestRegressor(random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)
    y_pred_rf_rounded = np.round(y_pred_rf)

    print("Random Forest - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf_rounded)))
    
    # Wizualizacja
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lin_rounded, alpha=0.5, label="Regresja liniowa", color="blue")
    plt.scatter(y_test, y_pred_rf_rounded, alpha=0.5, label="Random Forest", color="red")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Idealna linia")
    plt.xlabel("Rzeczywiste wartości")
    plt.ylabel("Przewidywane wartości")
    plt.legend()
    plt.title("Rzeczywiste vs przewidywane wartości", fontsize=16)
    plt.grid()
    plt.show()
    
    return lin_reg, rf_reg


# ---- Funkcja do grupowania ----
def cluster_data(X, n_clusters=3):
    # PCA do redukcji wymiarowości
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Grupowanie KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters_kmeans = kmeans.fit_predict(X)

    # Grupowanie hierarchiczne
    agg_clust = AgglomerativeClustering(n_clusters=n_clusters)
    clusters_agg = agg_clust.fit_predict(X)

    # Wizualizacja grup
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_kmeans, cmap="viridis", alpha=0.7)
    plt.title("K-Means - Grupowanie")
    plt.subplot(1, 2, 2)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters_agg, cmap="viridis", alpha=0.7)
    plt.title("Hierarchiczne grupowanie")
    plt.show()
    
    return clusters_kmeans, clusters_agg


# ---- Główna funkcja ----
if __name__ == "__main__":
    # Wczytanie danych
    file_path = "c:/Users/piotr/Desktop/Visualisation - Python/winequality-white.csv"
    data = load_and_clean_data(file_path)

    # Punkt 1: Eksploracyjna analiza danych
    visualize_data_distribution(data)

    # Punkt 2: Przygotowanie danych
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trenowanie klasyfikatorów: Drzewo decyzyjne oraz k-NN z wizualizacją
    tree_model, knn_model = train_classifiers(X_train, X_test, y_train, y_test)

    # Krzywe uczenia do porównania
    plot_learning_curves({"Drzewo Decyzyjne": tree_model, "k-NN": knn_model}, X_train, y_train)

    # Trenowanie regresorów oraz ich wizualizacja
    lin_reg, rf_reg = train_regressors(X_train, X_test, y_train, y_test)

    # Grupowanie na podstawie PCA + KMeans
    clusters_kmeans, clusters_agg = cluster_data(X)