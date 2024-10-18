import pandas as pd
import json
import matplotlib.pyplot as plt

def load_data(json_file_path):
    """Загружает данные из JSON файла и возвращает DataFrame."""
    with open(json_file_path, 'r') as f:
        data_json = json.load(f)
    data = pd.DataFrame(data_json)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

def calculate_bollinger_bands(data, window=20, k=2):
    """Вычисляет Полосы Боллинджера и добавляет их в DataFrame."""
    data['SMA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper'] = data['SMA'] + (k * data['STD'])
    data['Lower'] = data['SMA'] - (k * data['STD'])
    return data

def generate_signals(data):
    """Генерирует торговые сигналы на основе Полос Боллинджера."""
    data['Signal'] = 0
    data.loc[data['Close'] < data['Lower'], 'Signal'] = 1   # Сигнал на покупку
    data.loc[data['Close'] > data['Upper'], 'Signal'] = -1  # Сигнал на продажу
    data['Position'] = data['Signal'].replace(to_replace=0, value=pd.NA).ffill().fillna(0)
    return data

def calculate_returns(data):
    """Вычисляет доходность стратегии."""
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1)
    data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    return data

def plot_bollinger_bands(data):
    """Строит и отображает график Полос Боллинджера."""
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Close'], label='Цена закрытия')
    plt.plot(data.index, data['Upper'], label='Верхняя полоса', linestyle='--')
    plt.plot(data.index, data['Lower'], label='Нижняя полоса', linestyle='--')
    plt.fill_between(data.index, data['Upper'], data['Lower'], color='grey', alpha=0.1)
    plt.legend()
    plt.title('Полосы Боллинджера')
    plt.show()

def plot_strategy_performance(data):
    """Строит и отображает график эффективности стратегии."""
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Cumulative_Returns'], label='Кумулятивная доходность стратегии')
    plt.legend()
    plt.title('Эффективность стратегии')
    plt.show()

def generate_recommendation(data):
    """Анализирует последние данные и выдает рекомендацию покупать, продавать или удерживать позицию."""
    latest_data = data.iloc[-1]
    if latest_data['Close'] < latest_data['Lower']:
        recommendation = "Рекомендация: ПОКУПАТЬ. Цена ниже нижней полосы Боллинджера."
    elif latest_data['Close'] > latest_data['Upper']:
        recommendation = "Рекомендация: ПРОДАВАТЬ. Цена выше верхней полосы Боллинджера."
    else:
        recommendation = "Рекомендация: УДЕРЖИВАТЬ. Цена внутри полос Боллинджера."
    return recommendation

def main():
    json_file_path = 'data.json'  # Замените на путь к вашему JSON файлу
    data = load_data(json_file_path)
    data = calculate_bollinger_bands(data)
    data = generate_signals(data)
    data = calculate_returns(data)
    plot_bollinger_bands(data)
    plot_strategy_performance(data)

    # Получаем и выводим рекомендацию
    recommendation = generate_recommendation(data)
    print(recommendation)

if __name__ == '__main__':
    main()