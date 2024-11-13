import requests
import pandas
import datetime
import os
import re as regular
import datetime
"""import data_collector"""
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует

# Строчный класс, содержащий дневные уровни
"""class todaySupRes:
    def __init__(self, pivot, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3):
        self.pivot = pivot
        self.resistance_1 = resistance_1
        self.resistance_2 = resistance_2
        self.resistance_3 = resistance_3
        self.support_1 = support_1
        self.support_2 = support_2
        self.support_3 = support_3

    def __repr__(self):
        return f"Рассчитанные уровни:\nPivot Point: {self.pivot}\nResistance 1: {self.resistance_1}\nResistance 2: {self.resistance_1}\nResistance 3: {self.resistance_1}\nSupport 1: {self.support_1}\nSupport 2: {self.support_2}\nSupport 3: {self.support_3}"
        """


def today_levels(dataFrame: pandas.DataFrame) -> APK.todaySupRes:
    """
    Рассчитывает дневные уровни поддержки и сопротивления для переданного DataFrame.

    Аргументы:
    dataFrame -- pandas DataFrame с свечами
    
    Возвращает:
    Объект класса todaySupRes
    """
    #фильтрация фрейма по сегодняшним свечам
    now = datetime.datetime.now().date()
    now_data = dataFrame.loc[dataFrame['begin'] >= now.strftime("%Y-%m-%d")]

    #проаерка на пустоту
    if now_data.empty: raise APK.DatabaseError(f"dataframe is empty")

    #параметры
    high = now_data['high'].max()
    low = now_data['low'].min()
    close = now_data['close'].iloc[-1]

    #вычисления
    pivot = (high + low + close) / 3
    resistance_1 = 2 * pivot - low
    resistance_2 = pivot + (high - low)
    resistance_3 = pivot + 2 * (high - low)
    support_1 = 2 * pivot - high
    support_2 = pivot - (high - low)
    support_3 = pivot - 2 * (high - low)

    levels_class = APK.todaySupRes(pivot, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3)
    return levels_class


if __name__ == "__main__":
    # Чтение данных из файла
    data = pandas.read_json("storage\MOEX_2024-11-12_1D_[191120].json")
    
    # Проверка наличия необходимого столбца 'begin'
    if 'begin' not in data.columns:
        raise ValueError("Data does not contain 'close' column.")
    
    print(today_levels(data))
    print(data)
