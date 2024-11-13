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
        if not all('close' in item and 'volume' in item for item in data):
            raise APK.InvalidInputError("Некоторые элементы данных не содержат обязательных ключей 'close' или 'volume'.")
        return data
    except FileNotFoundError:
        raise APK.DatabaseError("Файл данных не найден.")
    except json.JSONDecodeError:
        raise APK.InvalidInputError("Ошибка при чтении JSON-файла.")

def find_support_resistance(data, lookback=20):
    """Находит уровни поддержки и сопротивления по цене закрытия за заданный период."""
    try:
        prices = [item['close'] for item in data[-lookback:]]
        support = min(prices)
        resistance = max(prices)
        return APK.todaySupRes(
            pivot=(support + resistance) / 2,
            resistance_1=resistance,
            resistance_2=resistance * 1.05,
            resistance_3=resistance * 1.1,
            support_1=support,
            support_2=support * 0.95,
            support_3=support * 0.9
        )
    except IndexError:
        raise APK.InvalidInputError("Недостаточно данных для анализа.")
    except KeyError:
        raise APK.InvalidInputError("В данных отсутствует ключ 'close'.")

def volume_strategy(data, support, resistance):
    """Применяет стратегию анализа объема на основе уровня поддержки и сопротивления."""
    signals = []

    for i in range(1, len(data)):
        current_price = data[i]['close']
        prev_price = data[i - 1]['close']
        current_volume = data[i]['volume']
        prev_volume = data[i - 1]['volume']
        
        if current_price > resistance and current_volume > prev_volume:
            signals.append((data[i]['date'], "Покупка", "Прорыв сопротивления с высоким объемом"))
        elif current_price < support and current_volume > prev_volume:
            signals.append((data[i]['date'], "Продажа", "Прорыв поддержки с высоким объемом"))
        elif current_price > prev_price and current_volume < prev_volume:
            signals.append((data[i]['date'], "Предупреждение", "Рост цены при снижающемся объеме"))
        elif current_price < prev_price and current_volume < prev_volume:
            signals.append((data[i]['date'], "Предупреждение", "Падение цены при снижающемся объеме"))

    return signals

def save_signals_to_file(signals, filename="signals_output.txt"):
    """Сохраняет сигналы в текстовый файл."""
    with open(filename, 'w') as file:
        for signal in signals:
            file.write(f"Дата: {signal[0]}, Сигнал: {signal[1]}, Описание: {signal[2]}\n")

if __name__ == "__main__":
    try:
        # Чтение данных из файла
        data = load_data(os.path.join(DATA_DIR, "FEES_2024-11-10_3D_[183522].json"))
        
        # Нахождение уровней поддержки и сопротивления
        sup_res = find_support_resistance(data)
        print(sup_res)

        # Выполнение стратегии анализа объема
        signals = volume_strategy(data, sup_res.support_1, sup_res.resistance_1)

        # Печать и сохранение сигналов
        for signal in signals:
            print(f"Дата: {signal[0]}, Сигнал: {signal[1]}, Описание: {signal[2]}")
        save_signals_to_file(signals)

        print("Сигналы успешно сохранены в файл 'signals_output.txt'.")
    except APK.ApplicationError as e:
        print(f"Ошибка: {e}")
