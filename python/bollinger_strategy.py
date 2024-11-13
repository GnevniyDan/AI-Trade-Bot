import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует

def load_data(filename):
    """Загружает данные из JSON-файла и конвертирует их в DataFrame."""
    try:
        data = pd.read_json(filename)
        required_columns = ['close', 'high', 'low']
        if not all(col in data.columns for col in required_columns):
            raise APK.InvalidInputError("Некоторые обязательные столбцы (close, high, low) отсутствуют.")
        return data
    except FileNotFoundError:
        raise APK.DatabaseError("Файл данных не найден.")
    except ValueError as e:
        raise APK.InvalidInputError(f"Ошибка при чтении файла: {e}")

def calculate_bollinger_bands(data, window=20, k=2):
    """Вычисляет Полосы Боллинджера и добавляет их в DataFrame."""
    data['SMA'] = data['close'].rolling(window=window).mean()
    data['STD'] = data['close'].rolling(window=window).std()
    data['Upper'] = data['SMA'] + (k * data['STD'])
    data['Lower'] = data['SMA'] - (k * data['STD'])
    return data

def generate_signals(data):
    """Генерирует торговые сигналы на основе Полос Боллинджера."""
    data['Signal'] = 0
    data.loc[data['close'] < data['Lower'], 'Signal'] = 1   # Сигнал на покупку
    data.loc[data['close'] > data['Upper'], 'Signal'] = -1  # Сигнал на продажу
    data['Position'] = data['Signal'].replace(to_replace=0, value=pd.NA).ffill().fillna(0)
    return data

def calculate_returns(data):
    """Вычисляет доходность стратегии."""
    data['Returns'] = data['close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1)
    data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    return data

def save_signals_to_file(data, filename="bollinger_signals_output.txt"):
    """Сохраняет сигналы и рекомендации в файл."""
    with open(filename, 'w') as file:
        for i in range(len(data)):
            if data['Signal'].iloc[i] != 0:
                file.write(f"Дата: {data.index[i]}, Сигнал: {'Покупка' if data['Signal'].iloc[i] == 1 else 'Продажа'}\n")
        file.write("\nАнализ последних данных:\n")
        recommendation = generate_recommendation(data)
        file.write(recommendation + '\n')

def generate_recommendation(data):
    """Анализирует последние данные и выдает рекомендацию."""
    latest_data = data.iloc[-1]
    if latest_data['close'] < latest_data['Lower']:
        recommendation = "Рекомендация: ПОКУПАТЬ. Цена ниже нижней полосы Боллинджера."
    elif latest_data['close'] > latest_data['Upper']:
        recommendation = "Рекомендация: ПРОДАВАТЬ. Цена выше верхней полосы Боллинджера."
    else:
        recommendation = "Рекомендация: УДЕРЖИВАТЬ. Цена внутри полос Боллинджера."
    return recommendation

def plot_bollinger_bands(data):
    """Строит график Полос Боллинджера."""
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['close'], label='Цена закрытия')
    plt.plot(data.index, data['Upper'], label='Верхняя полоса', linestyle='--')
    plt.plot(data.index, data['Lower'], label='Нижняя полоса', linestyle='--')
    plt.fill_between(data.index, data['Upper'], data['Lower'], color='grey', alpha=0.1)
    plt.legend()
    plt.title('Полосы Боллинджера')
    plt.show()

def plot_strategy_performance(data):
    """Строит график эффективности стратегии."""
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Cumulative_Returns'], label='Кумулятивная доходность стратегии')
    plt.legend()
    plt.title('Эффективность стратегии')
    plt.show()

if __name__ == "__main__":
    try:
        # Загрузка данных
        data = load_data(os.path.join(DATA_DIR, "FEES_2024-11-10_3D_[183522].json"))

        # Расчет индикаторов и генерация сигналов
        data = calculate_bollinger_bands(data)
        data = generate_signals(data)
        data = calculate_returns(data)

        # Печать и сохранение сигналов
        print("\nПоследние сигналы:")
        for i in range(len(data)):
            if data['Signal'].iloc[i] != 0:
                print(f"Дата: {data.index[i]}, Сигнал: {'Покупка' if data['Signal'].iloc[i] == 1 else 'Продажа'}")
        
        save_signals_to_file(data)
        print("Сигналы и рекомендации успешно сохранены в файл 'bollinger_signals_output.txt'.")

        # Визуализация результатов
        plot_bollinger_bands(data)
        plot_strategy_performance(data)

        # Печать рекомендации
        recommendation = generate_recommendation(data)
        print("\nАнализ последних данных:")
        print(recommendation)

        # Печать последних значений индикаторов
        print("\nПоследние значения индикаторов:")
        print(f"Цена закрытия: {data['close'].iloc[-1]:.2f}")
        print(f"Верхняя полоса: {data['Upper'].iloc[-1]:.2f}")
        print(f"Нижняя полоса: {data['Lower'].iloc[-1]:.2f}")
    except APK.ApplicationError as e:
        print(f"Ошибка: {e}")
