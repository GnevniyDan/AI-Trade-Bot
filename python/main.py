import _AppProjectKit as kit
import os
import pandas  # type: ignore
from colorama import Fore, Back, Style, init
from datetime import datetime, timedelta
import pytz
import time


#Московское время
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

"""Вновь разрабатываемые модули"""
from rsi_call import current_rsi_call
from bollinger_strategy import bollinger_strings
from candlestick_patterns import current_candlestick_patterns
from supres_levels import today_levels
from data_collector import ask_moex
from moving_averages import current_ma_analysis, get_ma_summary
from Stochastic_RSI import calculate_stochastic_rsi, get_stoch_rsi_summary
from volume import volume_analysis, get_volume_summary, find_support_resistance


# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует


#хз что это, но это нужно
pandas.set_option('future.no_silent_downcasting', True)


#баннер
with open("banner.txt", 'r', encoding="UTF-8") as banner_file:
    banner = banner_file.read()


#выбор анализируемого файла
data = pandas.read_json("storage\MOEX_2024-11-12_1D_[191120].json")


def processDataFrame(data: pandas.DataFrame) -> pandas.DataFrame:

    dataMod = data.copy(deep=True)

    #добавляем столбец с данными об индексе отн. силы
    dataMod["[RSI]"] = current_rsi_call(data, 18)

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

    #столбцы для скользящего среднего
    dataMod = current_ma_analysis(dataMod)

    #уровни поддержки и сопротивления
    levels = today_levels(data)

    #столбцы для объёмов
    dataMod = volume_analysis(dataMod, levels.support_1, levels.resistance_1)

    #столбцы для стохастик-рсай
    dataMod = calculate_stochastic_rsi(dataMod)

    #комментарии скользящего среднего
    summary_MA = get_ma_summary(dataMod)
    
    #комментарии для 100хахастика
    summary_SR = get_stoch_rsi_summary(dataMod)

    #комментарии для объёмов
    summary_V = get_volume_summary(dataMod)

    #рекоммендации
    Fore.YELLOW
    print(Fore.CYAN + "Мгновенные рекомендации:\n{}\n\n{}\n".format(bollinger.recommendation, levels))

    print("Сводная информация по Скользящим средним:")
    print(f"Тренд: {summary_MA['trend']}")
    print(f"Волатильность: {summary_MA['volatility_status']} ({summary_MA['current_volatility']:.2%})")
    print(f"Сигнал: {summary_MA['signal']}")
    print(f"Цена закрытия: {summary_MA['close']:.2f}")
    print(f"SMA: {summary_MA['sma']:.2f}")
    print(f"EMA: {summary_MA['ema']:.2f}\n")

    print("Текущее состояние по Стохастик-рсай:")
    print(f"Состояние: {summary_SR['condition']}")
    print(f"Тренд: {summary_SR['trend']}")
    print(f"RSI: {summary_SR['rsi']:.2f}")
    print(f"Стохастик %K: {summary_SR['stoch_k']:.2f}")
    print(f"Стохастик %D: {summary_SR['stoch_d']:.2f}\n")

    print("Текущее состояние по Объёмам:")
    print(f"Тренд объема: {summary_V['volume_trend']}")
    print(f"Текущий объем: {summary_V['current_volume']:,.0f}")
    print(f"Изменение объема: {summary_V['volume_change']:.2f}%")
    print(f"Сигнал ({summary_V['signal']}): {summary_V['signal_description']}\n" + Style.RESET_ALL)

    return dataMod.drop(columns=["RSI", "prev_close", "prev_volume"])


print(banner)
#columns_to_keep = ['begin', 'close', "RSI", "[B]top", "[B]bottom", "[B]cue", "[C]Hammer", '[C]HangingMan', '[C]Engulfing']
#print(dataMod.tail(17)[columns_to_keep])
print(processDataFrame(data).tail(17))