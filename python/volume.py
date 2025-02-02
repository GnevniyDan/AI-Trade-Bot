import json
import os
import pandas as pd
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_data(filename):
    """Загружает данные из JSON-файла в DataFrame."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Преобразование JSON в DataFrame
        df = pd.DataFrame(data)
        
        # Проверка наличия необходимых столбцов
        required_columns = ['close', 'volume', 'date']
        if not all(col in df.columns for col in required_columns):
            raise APK.InvalidInputError("Отсутствуют обязательные столбцы: 'close', 'volume' или 'date'")
        
        # Преобразование типов данных
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except FileNotFoundError:
        raise APK.DatabaseError("Файл данных не найден.")
    except json.JSONDecodeError:
        raise APK.InvalidInputError("Ошибка при чтении JSON-файла.")
    except ValueError as e:
        raise APK.InvalidInputError(f"Ошибка преобразования данных: {str(e)}")

def find_support_resistance(df, lookback=20):
    """Находит уровни поддержки и сопротивления используя pandas."""
    try:
        # Получаем последние n значений
        recent_prices = df['close'].tail(lookback)
        
        support = recent_prices.min()
        resistance = recent_prices.max()
        
        return APK.todaySupRes(
            pivot=(support + resistance) / 2,
            resistance_1=resistance,
            resistance_2=resistance * 1.05,
            resistance_3=resistance * 1.1,
            support_1=support,
            support_2=support * 0.95,
            support_3=support * 0.9
        )
    except Exception as e:
        raise APK.InvalidInputError(f"Ошибка при расчете уровней: {str(e)}")

def volume_strategy(df, support, resistance):
    """Применяет стратегию анализа объема используя pandas."""
    signals = []
    
    # Создаем сдвинутые значения для сравнения
    df['prev_close'] = df['close'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    
    # Пропускаем первую строку, так как для неё нет предыдущих значений
    for idx, row in df.iloc[1:].iterrows():
        if row['close'] > resistance and row['volume'] > row['prev_volume']:
            signals.append((row['date'], "Покупка", "Прорыв сопротивления с высоким объемом"))
        elif row['close'] < support and row['volume'] > row['prev_volume']:
            signals.append((row['date'], "Продажа", "Прорыв поддержки с высоким объемом"))
        elif row['close'] > row['prev_close'] and row['volume'] < row['prev_volume']:
            signals.append((row['date'], "Предупреждение", "Рост цены при снижающемся объеме"))
        elif row['close'] < row['prev_close'] and row['volume'] < row['prev_volume']:
            signals.append((row['date'], "Предупреждение", "Падение цены при снижающемся объеме"))
    
    return signals

def save_signals_to_file(signals, filename="signals_output.txt"):
    """Сохраняет сигналы в текстовый файл."""
    with open(filename, 'w', encoding='utf-8') as file:
        for signal in signals:
            file.write(f"Дата: {signal[0]}, Сигнал: {signal[1]}, Описание: {signal[2]}\n")

if __name__ == "__main__":
    try:
        # Чтение данных из файла
        filename = os.path.join(DATA_DIR, "FEES_2024-11-10_3D_[183522].json")
        df = load_data(filename)
        
        # Нахождение уровней поддержки и сопротивления
        sup_res = find_support_resistance(df)
        print(sup_res)
        
        # Выполнение стратегии анализа объема
        signals = volume_strategy(df, sup_res.support_1, sup_res.resistance_1)
        
        # Печать и сохранение сигналов
        for signal in signals:
            print(f"Дата: {signal[0]}, Сигнал: {signal[1]}, Описание: {signal[2]}")
            
        save_signals_to_file(signals)
        print("Сигналы успешно сохранены в файл 'signals_output.txt'.")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")
