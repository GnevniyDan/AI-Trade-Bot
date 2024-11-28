import _AppProjectKit as kit
import os
import pandas  # type: ignore
from colorama import Fore, Back, Style, init

"""Вновь разрабатываемые модули"""
from rsi_call import current_rsi_call
from bollinger_strategy import bollinger_strings
from candlestick_patterns import current_candlestick_patterns


# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует

#
pandas.set_option('future.no_silent_downcasting', True)

#
with open("banner.txt", 'r', encoding="UTF-8") as banner_file:
    banner = banner_file.read()



#
data = pandas.read_json("storage\MOEX_2024-11-12_1D_[191120].json")
dataMod = data.copy(deep=True)

#добавляем столбец с данными об индексе отн. силы
dataMod["RSI"] = current_rsi_call(data, 18)

#столбцы для боллинджера
bollinger = bollinger_strings(data)
dataMod["[B]top"] = bollinger.dataFrame["Lower"]
dataMod["[B]bottom"] = bollinger.dataFrame["Upper"]
dataMod["[B]cue"] = bollinger.dataFrame["Signal"] #спросить у Дани 

#столбцы для свечных паттернов
candlestick = current_candlestick_patterns(data)
dataMod["[C]Hammer"] = candlestick["Hammer"]
dataMod['[C]HangingMan'] = candlestick['HangingMan']
dataMod['[C]Engulfing'] = candlestick['Engulfing']

#столбцы для уровней поддержки и сопротивления


print(banner)
Fore.YELLOW
print(Fore.CYAN + "Мгновенные рекомендации:\n{}\n\n".format(bollinger.recommendation) + Style.RESET_ALL)
columns_to_keep = ['begin', 'close', 'Engulfing']
print(dataMod.tail(17)[])