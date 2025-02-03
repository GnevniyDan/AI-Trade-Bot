import _AppProjectKit as kit
import os
import pandas  # type: ignore
from colorama import Fore, Back, Style, init
from datetime import datetime, timedelta
import pytz
import time
import json

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


#Московское время
MOSCOW_TZ = pytz.timezone('Europe/Moscow')


def load_config():
    with open('CONFIG.json', 'r', encoding='utf-8') as file:
        return json.load(file)
    

#подгрузка конфига
CONFIG = load_config()
TICKER = CONFIG.get("ticker")
INTERVAL = CONFIG.get("interval")  # Интервал свечей в минутах


#выбор анализируемого файла
data = pandas.read_json("storage\YDEX_2025-02-02_1D_[193507].json")
print(data)


def processDataFrame(data: pandas.DataFrame) -> pandas.DataFrame:

    print(banner)

    #создание проанализированного фрейма
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

    print(dataMod.tail[7])

    return dataMod.drop(columns=["RSI", "prev_close", "prev_volume"])


def main():

    #добавление в видимость
    global TICKER
    global INTERVAL
    global data

    #временные границы торгового дня (установить нужные)
    market_open = datetime.now(MOSCOW_TZ).replace(hour=6, minute=50, second=0, microsecond=0) #6:50
    market_close = datetime.now(MOSCOW_TZ).replace(hour=22, minute=50, second=0, microsecond=0) #18:50
    
    while datetime.now(MOSCOW_TZ) < market_close:

        now = datetime.now(MOSCOW_TZ)

        if now >= market_open and now.minute % INTERVAL == 0 and now.second == 0 or now.second % 10 == 0:
            upcomingCandle = ask_moex(ticker=TICKER, interval=INTERVAL, period="1D", record=False).iloc[-1]
            data.iloc[-1] = upcomingCandle
            processDataFrame(data)
            time.sleep(10)  # Короткая пауза, чтобы избежать многократного вызова в одну секунду (так чат сказал)


if __name__ == "__main__":
    main()