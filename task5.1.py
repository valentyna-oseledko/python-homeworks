# Імпорт необхідних бібліотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Встановлення стилю для графіків
sns.set(style='whitegrid')

# Завантаження датасету
df = pd.read_csv('Mall_Customers.csv')

# Первинний аналіз даних
print("Перших 5 рядків таблиці:")
print(df.head())

print("\nІнформація про датасет:")
print(df.info())

print("\nОписова статистика:")
print(df.describe())

# Перевірка на пропущені значення
print("\nПропущені значення у кожному стовпчику:")
print(df.isnull().sum())

# Побудова гістограм для кожної змінної
df.hist(bins=20, figsize=(15, 10))
plt.suptitle('Розподіл змінних')
plt.show()

# Розрахунок кореляційної матриці
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Кореляційна матриця')
plt.show()

# Стандартизація даних
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Створення нового DataFrame з стандартизованими даними
df_scaled = pd.DataFrame(df_scaled, columns=features)

# Метод ліктя (Elbow method)
inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Побудова графіку залежності інерції від кількості кластерів
plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Inertia')
plt.title('Elbow Method для визначення оптимальної кількості кластерів')
plt.grid(True)
plt.show()

# Розрахунок коефіцієнта силуету
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    silhouette_scores.append(silhouette_score(df_scaled, labels))

# Побудова графіку залежності коефіцієнта силуету від кількості кластерів
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score для визначення оптимальної кількості кластерів')
plt.grid(True)
plt.show()

# Вибір оптимальної кількості кластерів (припустимо, це 5 на основі графіків)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Візуалізація результатів кластеризації
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=300, c='red', marker='X', label='Центроїди')
plt.title('Результати кластеризації (K-means)')
plt.legend()
plt.grid(True)
plt.show()

# Аналіз кожного кластера
cluster_analysis = df.groupby('Cluster').mean()
print("\nСередні значення показників для кожного кластера:")
print(cluster_analysis)

# Маркетингові стратегії для кожного сегмента
for cluster in range(optimal_clusters):
    print(f"\nКластер {cluster}:")
    print(df[df['Cluster'] == cluster].describe())

# Додаткова кластеризація методом DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(df_scaled)

# Візуалізація результатів DBSCAN
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='DBSCAN_Cluster', data=df, palette='coolwarm')
plt.title('DBSCAN Кластеризація')
plt.grid(True)
plt.show()

# Додаткова кластеризація методом ієрархічної кластеризації
agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters)
df['Agglo_Cluster'] = agg_clustering.fit_predict(df_scaled)

# Візуалізація результатів ієрархічної кластеризації
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Agglo_Cluster', data=df, palette='Set1')
plt.title('Ієрархічна Кластеризація')
plt.grid(True)
plt.show()
