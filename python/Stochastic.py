import json
import os
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует

def load_data(filename):
    """Загружает данные из JSON-файла."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        # Проверка всех элементов на наличие необходимых ключей
        required_columns = ['high', 'low', 'close']
        if not all(all(col in item for col in required_columns) for item in data):
            raise APK.InvalidInputError("Некоторые элементы данных не содержат обязательных ключей (high, low, close).")
        return data
    except FileNotFoundError:
        raise APK.DatabaseError("Файл данных не найден.")
    except json.JSONDecodeError:
        raise APK.InvalidInputError("Ошибка при чтении JSON-файла.")

def calculate_stochastic(data, period=14):
    """Рассчитывает стохастический осциллятор за заданный период."""
    stochastic_values = []

    for i in range(period, len(data)):
        highest_high = max(item['high'] for item in data[i-period:i])
        lowest_low = min(item['low'] for item in data[i-period:i])
        close = data[i]['close']

        if highest_high != lowest_low:  # Защита от деления на ноль
            stochastic = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        else:
            stochastic = 50  # Среднее значение в случае отсутствия движения

        stochastic_values.append({
            'date': data[i]['date'],
            'stochastic': stochastic
        })
    
    return stochastic_values

def stochastic_strategy(stochastic_values):
    """Применяет торговую стратегию на основе значений стохастического осциллятора."""
    signals = []

    for item in stochastic_values:
        if item['stochastic'] > 80:
            signals.append((item['date'], "Продажа", "Перекупленность: Стохастик выше 80"))
        elif item['stochastic'] < 20:
            signals.append((item['date'], "Покупка", "Перепроданность: Стохастик ниже 20"))

    return signals

def save_signals_to_file(signals, filename="stochastic_signals_output.txt"):
    """Сохраняет сигналы в текстовый файл."""
    with open(filename, 'w') as file:
        for signal in signals:
            file.write(f"Дата: {signal[0]}, Сигнал: {signal[1]}, Описание: {signal[2]}\n")

if __name__ == "__main__":
    try:
        # Загрузка данных из файла
        data = load_data(os.path.join(DATA_DIR, "MOEX_2024-11-12_1D_[191120].json"))
        
        # Расчет значений стохастического осциллятора
        stochastic_values = calculate_stochastic(data)
        
        # Получение торговых сигналов
        signals = stochastic_strategy(stochastic_values)
        
        # Печать и сохранение последних 50 сигналов
        print("\nПоследние сигналы:")
        for signal in signals[-50:]:
            print(f"Дата: {signal[0]}, Сигнал: {signal[1]}, Описание: {signal[2]}")
        
        save_signals_to_file(signals)
        print("Сигналы успешно сохранены в файл 'stochastic_signals_output.txt'.")
    except APK.ApplicationError as e:
        print(f"Ошибка: {e}")
