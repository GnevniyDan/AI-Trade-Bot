import requests
import pandas
import datetime
import os
import re as regular
import datetime
import _AppProjectKit as APK


#быстрые проверки трендов, window - количество свеч
def is_downtrend(dataFrame, index, window):
    if index < window:
        return False
    return dataFrame['close'].iloc[index - window:index].is_monotonic_decreasing


def is_uptrend(dataFrame, index, window):
    if index < window:
        return False
    return dataFrame['close'].iloc[index - window:index].is_monotonic_increasing


# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует


# Функция для идентификации паттерна "Молот"
def is_hammer(row, dataFrame, index, window):
    body = abs(row['close'] - row['open'])
    lower_shadow = row['open'] - row['low'] if row['close'] > row['open'] else row['close'] - row['low']
    upper_shadow = row['high'] - max(row['close'], row['open'])
    return lower_shadow > body * 2 and upper_shadow < body and is_downtrend(dataFrame, index, window)


def is_hanging_man(row, dataFrame, index, window):
    body = abs(row['close'] - row['open'])
    lower_shadow = row['open'] - row['low'] if row['close'] > row['open'] else row['close'] - row['low']
    upper_shadow = row['high'] - max(row['close'], row['open'])
    return lower_shadow > body * 2 and upper_shadow < body and is_uptrend(dataFrame, index, window)


def is_engulfing(df, index):
    if index == 0:
        return False  # Поглощение невозможно на первой свече
    
    prev_row = df.iloc[index - 1]
    curr_row = df.iloc[index]
    
    # Бычье поглощение
    if prev_row['close'] < prev_row['open'] and curr_row['close'] > curr_row['open'] and \
       curr_row['open'] < prev_row['close'] and curr_row['close'] > prev_row['open']:
        return 'Bullish Engulfing'
    
    # Медвежье поглощение
    if prev_row['close'] > prev_row['open'] and curr_row['close'] < curr_row['open'] and \
       curr_row['open'] > prev_row['close'] and curr_row['close'] < prev_row['open']:
        return 'Bearish Engulfing'
    
    return False


def current_candlestick_patterns(dataFrame: pandas.DataFrame, window: int = 3) -> pandas.DataFrame:
    """
    Модифицирует переданный свечной фрейм, добавляя к нему колонки с булевыми флажками,
    помечающими наличие того или иного паттерна.
    «Молот» (Hammer) должен появляться после нисходящего тренда и сигнализировать о развороте вверх.
    «Повешенный» (Hanging Man) появляется после восходящего тренда и сигнализирует о развороте вниз.

    Бычье поглощение (Bullish Engulfing):
    Описание: Этот паттерн возникает после нисходящего тренда и состоит из двух свечей. Первая свеча – медвежья (с понижением), а вторая – бычья (с повышением), которая полностью охватывает тело первой.
    Сигналы: Указывает на сильный сигнал разворота вверх, особенно если вторая свеча закрывается выше первого тела.

    Медвежье поглощение (Bearish Engulfing):
    Описание: Появляется на вершине восходящего тренда и также состоит из двух свечей. Первая – бычья, а вторая – медвежья, которая полностью перекрывает тело первой.
    Сигналы: Сильный сигнал разворота вниз, особенно при закрытии второй свечи ниже тела первой.
    """
    #преобразование фрейма:
    dataFrame['Hammer'] = False
    dataFrame['HangingMan'] = False
    dataFrame['Engulfing'] = False

    #расстановка флажков
    dataFrame['Hammer'] = [is_hammer(row, dataFrame, i, window) for i, row in dataFrame.iterrows()]
    dataFrame['HangingMan'] = [is_hanging_man(row, dataFrame, i, window) for i, row in dataFrame.iterrows()]
    #Не более 100 шгагов назад
    dataFrame['Engulfing'] = [is_engulfing(dataFrame, i) for i in range(max(len(dataFrame), 100))]

    columns_to_keep = ['Hammer', 'HangingMan', 'Engulfing']

    return dataFrame[columns_to_keep]


if __name__ == "__main__":
    # Чтение данных из файла
    data = pandas.read_json("storage\FEES_2024-11-10_3D_[183522].json")
    
    # Проверка наличия необходимого столбца 'close'
    if data.empty:
        raise ValueError("database is empty")
    
    # Вывод последних 10 значений RSI
    print(current_candlestick_patterns(data, 4).tail(50))
