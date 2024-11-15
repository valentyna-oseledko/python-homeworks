# Імпорт необхідних бібліотек
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import find_peaks

# Встановлення стилю для графіків
sns.set(style='whitegrid')

# Частина 1: Підготовка та візуалізація даних
# Завантаження історичних даних котирувань акцій за останній рік
ticker = 'AAPL'  # Оберіть тікер для аналізу
data = yf.download(ticker, period='1y')

# Первинний аналіз даних
print("Перших 5 рядків даних:")
print(data.head())

# Перевірка на пропущені значення
print("\nПеревірка на пропущені значення:")
print(data.isnull().sum())

# Побудова графіку зміни ціни закриття
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Ціна закриття', color='blue')
plt.title(f'Зміна ціни закриття акцій {ticker}')
plt.xlabel('Дата')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid(True)
plt.show()

# Базова описова статистика
print("\nОписова статистика:")
print(data['Close'].describe())

# Частина 2: Аналіз компонентів часового ряду
# Виконання декомпозиції часового ряду
decompose_result = seasonal_decompose(data['Close'], model='additive', period=30)

# Виділення тренду, сезонної та випадкової компоненти
trend = decompose_result.trend
seasonal = decompose_result.seasonal
residual = decompose_result.resid

# Візуалізація компонентів
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(data['Close'], label='Оригінальний ряд')
plt.legend()
plt.subplot(412)
plt.plot(trend, label='Тренд', color='orange')
plt.legend()
plt.subplot(413)
plt.plot(seasonal, label='Сезонність', color='green')
plt.legend()
plt.subplot(414)
plt.plot(residual, label='Випадкова компонента', color='red')
plt.legend()
plt.tight_layout()
plt.show()

# Частина 3: Технічний аналіз
# Розрахунок простих ковзних середніх (7 та 30 днів)
data['SMA_7'] = data['Close'].rolling(window=7).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()

# Розрахунок RSI
def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data['Close'])

# Розрахунок волатильності (30-денна стандартна девіація)
data['Volatility'] = data['Close'].rolling(window=30).std()

# Візуалізація технічних індикаторів
plt.figure(figsize=(14, 10))
plt.subplot(311)
plt.plot(data['Close'], label='Ціна закриття', color='blue')
plt.plot(data['SMA_7'], label='SMA 7', color='orange')
plt.plot(data['SMA_30'], label='SMA 30', color='green')
plt.legend()

plt.subplot(312)
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.legend()

plt.subplot(313)
plt.plot(data['Volatility'], label='Волатильність', color='brown')
plt.legend()
plt.tight_layout()
plt.show()

# Частина 4: Прогнозування
# Розділення даних на навчальну та тестову вибірки
train_size = int(len(data) * 0.8)
train, test = data['Close'][:train_size], data['Close'][train_size:]

# Прогноз за допомогою моделі експоненційного згладжування
model_es = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
fit_es = model_es.fit()
pred_es = fit_es.forecast(len(test))

# Прогноз за допомогою ARIMA моделі
model_arima = ARIMA(train, order=(5, 1, 0))
fit_arima = model_arima.fit()
pred_arima = fit_arima.forecast(len(test))

# Оцінка якості прогнозу
mse_es = mean_squared_error(test, pred_es)
mae_es = mean_absolute_error(test, pred_es)
mse_arima = mean_squared_error(test, pred_arima)
mae_arima = mean_absolute_error(test, pred_arima)

print(f"\nЕкспоненційне згладжування - MSE: {mse_es:.2f}, MAE: {mae_es:.2f}")
print(f"ARIMA модель - MSE: {mse_arima:.2f}, MAE: {mae_arima:.2f}")

# Візуалізація результатів прогнозування
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Навчальні дані')
plt.plot(test.index, test, label='Тестові дані', color='black')
plt.plot(test.index, pred_es, label='Прогноз (Експоненційне згладжування)', color='orange')
plt.plot(test.index, pred_arima, label='Прогноз (ARIMA)', color='green')
plt.title('Прогноз цін закриття акцій')
plt.xlabel('Дата')
plt.ylabel('Ціна ($)')
plt.legend()
plt.grid(True)
plt.show()
