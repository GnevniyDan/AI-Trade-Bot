import requests
import pandas
import datetime
import os
import re as regular
import data_collector

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует


def current_rsi_call(dataFrame: pandas.DataFrame, candle_frame: int = 18) -> pandas.Series:
    """
    Рассчитывает индекс относительной силы (RSI) для переданного DataFrame.

    Аргументы:
    dataFrame -- pandas DataFrame с данными, содержащими колонку 'close'
    candle_frame -- период для скользящего среднего (по умолчанию 18)

    Возвращает:
    pandas.Series с рассчитанным RSI
    """

    # Колонка с закрытиями
    close = dataFrame['close']

    # Проверка длины серии
    if close.size < candle_frame:
        raise data_collector.DatabaseError(f"Data frame is too short: {close.size} rows, requires at least {candle_frame}.")

    # [1, 3, 6, 10, 15] -> [2, 3, 4, 5]
    delta = close.diff()

    # Все значения, не подпадающие под условие, заменяются нулём
    positive = delta.where(delta > 0, 0)
    negative = delta.where(delta < 0, 0).abs()  # Приводим к положительным значениям

    # Скользящее среднее из pandas (катящееся mean)
    avg_pos = positive.rolling(window=candle_frame, min_periods=1).mean()
    avg_neg = negative.rolling(window=candle_frame, min_periods=1).mean()

    # Значение относительной силы
    rs = avg_pos / avg_neg

    # Индекс относительной силы (RSI)
    rsi = 100 - (100 / (1 + rs))

    return rsi


if __name__ == "__main__":
    # Чтение данных из файла
    data = pandas.read_json("storage/FEES_2024-10-10_3D_[214156].json")
    
    # Проверка наличия необходимого столбца 'close'
    if 'close' not in data.columns:
        raise ValueError("Data does not contain 'close' column.")
    
    # Вывод последних 10 значений RSI
    print(current_rsi_call(data).tail(10))