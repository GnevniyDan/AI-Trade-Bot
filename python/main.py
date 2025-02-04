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


#настройки библиотек
pandas.set_option('future.no_silent_downcasting', True)
init(autoreset=True)


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
FILENAME = CONFIG.get("filename")   #имя файла базы данных, к которой будут приписываться строчки
INSTANT_PROCESSING = CONFIG.get("instant processing") #bool один раз обработать массив без ожидания свечей
SELF_CREATION = CONFIG.get("self creation") #bool не использовать файл data а создать его с нуля 


#выбор анализируемого файла
combined_path = os.path.join(DATA_DIR, FILENAME)
if SELF_CREATION:
    data = ask_moex(ticker=TICKER, interval=INTERVAL, period="1D", record=True)
    print("MOEX answer catched!")
    FILENAME = "self_created_{}_{}.json".format(TICKER, INTERVAL)
else:
    data = pandas.read_json(combined_path)
print(Fore.BLACK + Back.WHITE + 'Последние свечи в базе\n', data.tail(4), "\n")


def processDataFrame(data: pandas.DataFrame) -> pandas.DataFrame:

    global TICKER
    global INTERVAL
    global MOSCOW_TZ

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

    print(Fore.CYAN + "Сводная информация по Скользящим средним:")
    print(Fore.CYAN + f"Тренд: {summary_MA['trend']}")
    print(Fore.CYAN + f"Волатильность: {summary_MA['volatility_status']} ({summary_MA['current_volatility']:.2%})")
    print(Fore.CYAN + f"Сигнал: {summary_MA['signal']}")
    print(Fore.CYAN + f"Цена закрытия: {summary_MA['close']:.2f}")
    print(Fore.CYAN + f"SMA: {summary_MA['sma']:.2f}")
    print(Fore.CYAN + f"EMA: {summary_MA['ema']:.2f}\n")

    print(Fore.CYAN + "Текущее состояние по Стохастик-рсай:")
    print(Fore.CYAN + f"Состояние: {summary_SR['condition']}")
    print(Fore.CYAN + f"Тренд: {summary_SR['trend']}")
    print(Fore.CYAN + f"RSI: {summary_SR['rsi']:.2f}")
    print(Fore.CYAN + f"Стохастик %K: {summary_SR['stoch_k']:.2f}")
    print(Fore.CYAN + f"Стохастик %D: {summary_SR['stoch_d']:.2f}\n")

    print(Fore.CYAN + "Текущее состояние по Объёмам:")
    print(Fore.CYAN + f"Тренд объема: {summary_V['volume_trend']}")
    print(Fore.CYAN + f"Текущий объем: {summary_V['current_volume']:,.0f}")
    print(Fore.CYAN + f"Изменение объема: {summary_V['volume_change']:.2f}%")
    print(Fore.CYAN + f"Сигнал ({summary_V['signal']}): {summary_V['signal_description']}\n" + Style.RESET_ALL)

    print(Fore.BLACK + Back.WHITE + 'Анализ {}-минутных свечей по тикеру {} в {}\n'.format(INTERVAL, TICKER, datetime.now(MOSCOW_TZ)), dataMod.tail(4))

    return dataMod.drop(columns=["RSI", "prev_close", "prev_volume"])


def main():

    #добавление в видимость
    global TICKER
    global INTERVAL
    global data
    global INSTANT_PROCESSING
    global DATA_DIR

    if INSTANT_PROCESSING:
        #print(os.path.join(DATA_DIR, os.path.splitext(FILENAME)[0], "_postprocessed.json"))
        processDataFrame(data).to_json(os.path.join(DATA_DIR, os.path.splitext(FILENAME)[0] + "_processed.json"))
        os._exit(1)

    #временные границы торгового дня (установить нужные)
    market_open = datetime.now(MOSCOW_TZ).replace(hour=0, minute=10, second=0, microsecond=0) #6:50
    market_close = datetime.now(MOSCOW_TZ).replace(hour=23, minute=50, second=0, microsecond=0) #18:50
    
    while datetime.now(MOSCOW_TZ) < market_close:

        now = datetime.now(MOSCOW_TZ)

        if now >= market_open and now.minute % INTERVAL == 0 and now.second == 0:

            upcomingCandle = ask_moex(ticker=TICKER, interval=INTERVAL, period="1D", record=False).iloc[-1]
            data.loc[len(data)] = upcomingCandle

            if upcomingCandle.empty:
                raise  kit.DatabaseError("current candle is empty")
            
            processDataFrame(data).to_json(os.path.join(DATA_DIR, os.path.splitext(FILENAME)[0] + "_processed.json"))
            time.sleep(10)  # Короткая пауза, чтобы избежать многократного вызова в одну секунду (так чат сказал)


if __name__ == "__main__":
    main()