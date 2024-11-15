# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Встановлення стилю для графіків
sns.set(style='whitegrid')

# Завантаження датасету
df = pd.read_csv('Mall_Customers.csv')

# Частина 1: Попередній аналіз даних (EDA)
print("Перших 5 рядків таблиці:")
print(df.head())

print("\nІнформація про датасет:")
print(df.info())

print("\nОписова статистика:")
print(df.describe())

# Перевірка на пропущені значення
print("\nПропущені значення у кожному стовпчику:")
print(df.isnull().sum())

# Кодування категоріальної змінної (Gender)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male=1, Female=0

# Стандартизація числових ознак
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Створення нового DataFrame з стандартизованими даними
df_scaled = pd.DataFrame(df_scaled, columns=features)

# Частина 2: Застосування методу PCA
# Виконання PCA
pca = PCA()
df_pca = pca.fit_transform(df_scaled)

# Визначення кількості компонент, що пояснюють 95% дисперсії
explained_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.argmax(explained_variance >= 0.95) + 1
print(f"\nОптимальна кількість головних компонент (для 95% дисперсії): {optimal_components}")

# Візуалізація результатів PCA у 2D
pca_2d = PCA(n_components=2)
df_pca_2d = pca_2d.fit_transform(df_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(df_pca_2d[:, 0], df_pca_2d[:, 1], c='blue', s=50)
plt.title('PCA (2 компоненти)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

# Візуалізація результатів PCA у 3D
pca_3d = PCA(n_components=3)
df_pca_3d = pca_3d.fit_transform(df_scaled)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_3d[:, 0], df_pca_3d[:, 1], df_pca_3d[:, 2], c='green', s=50)
ax.set_title('PCA (3 компоненти)')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()

# Частина 3: Застосування t-SNE
# Виконання t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

# Візуалізація результатів t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c='purple', s=50)
plt.title('t-SNE (2 компоненти)')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.grid(True)
plt.show()

# Частина 4: Кластеризація K-means на PCA та t-SNE даних
# Кластеризація на PCA-даних
kmeans_pca = KMeans(n_clusters=5, random_state=42)
clusters_pca = kmeans_pca.fit_predict(df_pca_2d)

# Візуалізація кластерів на PCA-даних
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca_2d[:, 0], y=df_pca_2d[:, 1], hue=clusters_pca, palette='viridis', s=100)
plt.title('Кластеризація K-means на PCA даних')
plt.grid(True)
plt.show()

# Кластеризація на t-SNE даних
kmeans_tsne = KMeans(n_clusters=5, random_state=42)
clusters_tsne = kmeans_tsne.fit_predict(df_tsne)

# Візуалізація кластерів на t-SNE даних
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_tsne[:, 0], y=df_tsne[:, 1], hue=clusters_tsne, palette='coolwarm', s=100)
plt.title('Кластеризація K-means на t-SNE даних')
plt.grid(True)
plt.show()

# Порівняння результатів кластеризації
silhouette_pca = silhouette_score(df_pca_2d, clusters_pca)
silhouette_tsne = silhouette_score(df_tsne, clusters_tsne)
print(f"\nSilhouette Score для PCA: {silhouette_pca:.2f}")
print(f"Silhouette Score для t-SNE: {silhouette_tsne:.2f}")

# Аналіз результатів
print("\nАналіз кластерів на основі PCA:")
print(df.assign(Cluster=clusters_pca).groupby('Cluster').mean())

print("\nАналіз кластерів на основі t-SNE:")
print(df.assign(Cluster=clusters_tsne).groupby('Cluster').mean())

# Рекомендації щодо маркетингових стратегій
print("\nМаркетингові стратегії для кожного кластера:")
for i in range(5):
    print(f"\nКластер {i}:")
    print(df[df['Cluster'] == i].describe())
