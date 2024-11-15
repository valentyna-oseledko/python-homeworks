# Імпорт бібліотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# 1. Функція для завантаження та попереднього огляду даних
def load_data():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target
    return df


# 2. Функція для проведення дослідницького аналізу даних (EDA)
def exploratory_data_analysis(df):
    print("Описова статистика:")
    print(df.describe())

    print("\nПеревірка на пропущені значення:")
    print(df.isnull().sum())

    print("\nТипи даних:")
    print(df.dtypes)

    # Візуалізація розподілу ознак
    df.hist(bins=20, figsize=(15, 10))
    plt.suptitle('Розподіл ознак')
    plt.show()

    # Boxplot для виявлення викидів
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title('Boxplot для виявлення викидів')
    plt.show()

    # Кореляційна матриця
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Кореляційна матриця')
    plt.show()


# 3. Функція для підготовки даних
def prepare_data(df):
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабування даних
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# 4. Функція для побудови та оцінки моделі
def build_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Проста лінійна регресія (тільки MedInc)
    simple_model = LinearRegression()
    simple_model.fit(X_train[:, [0]], y_train)
    y_pred_simple = simple_model.predict(X_test[:, [0]])
    mse_simple = mean_squared_error(y_test, y_pred_simple)
    r2_simple = r2_score(y_test, y_pred_simple)
    
    print(f"\nПроста лінійна регресія (MedInc) - MSE: {mse_simple:.2f}, R²: {r2_simple:.2f}")
    
    # Множинна лінійна регресія (всі ознаки)
    multi_model = LinearRegression()
    multi_model.fit(X_train, y_train)
    y_pred_multi = multi_model.predict(X_test)
    mse_multi = mean_squared_error(y_test, y_pred_multi)
    r2_multi = r2_score(y_test, y_pred_multi)
    
    print(f"Множинна лінійна регресія - MSE: {mse_multi:.2f}, R²: {r2_multi:.2f}")
    
    # Графік передбачених vs реальних значень
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_multi, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices (Множинна регресія)')
    plt.grid(True)
    plt.show()
    
    return multi_model


# 5. Функція для прогнозування на основі нових даних
def predict_price(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]


# Основна програма
if __name__ == "__main__":
    df = load_data()
    exploratory_data_analysis(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)
    model = build_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Прогноз для нового будинку
    new_house = [8.0, 25, 6.5, 1.0, 3000, 3.0, 34.19, -118.5]
    price = predict_price(model, scaler, new_house)
    print(f"\nПрогнозована ціна для нового будинку: ${price * 100000:.2f}")
