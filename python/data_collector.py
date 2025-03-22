import requests
import apimoex
import pandas as pd
import datetime
import os 
import re
import json
import time
from typing import Optional, Dict, Any, Union, Tuple

import _AppProjectKit as APK

# Константы
# =========
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Кэш для хранения запросов, чтобы избежать повторного запроса одних и тех же данных
# Ключ: (ticker, interval, start_date, end_date), Значение: (timestamp, data)
REQUEST_CACHE: Dict[Tuple[str, int, str, Optional[str]], Tuple[float, pd.DataFrame]] = {}
CACHE_EXPIRY = 300  # 5 минут в секундах
MAX_RETRIES = 3     # Максимальное количество попыток запроса к API


def is_cache_valid(cache_key: Tuple[str, int, str, Optional[str]]) -> bool:
    """
    Проверяет, действителен ли кэш для данного ключа.
    
    Аргументы:
        cache_key: Ключ кэша (ticker, interval, start_date, end_date)
        
    Возвращает:
        bool: True, если кэш действителен, иначе False
    """
    if cache_key not in REQUEST_CACHE:
        return False
    
    timestamp, _ = REQUEST_CACHE[cache_key]
    return (time.time() - timestamp) < CACHE_EXPIRY


def ask_moex(ticker: str = "FEES", 
             interval: int = 10, 
             period: str = "1D", 
             end: Optional[str] = None, 
             record: bool = False,
             force_refresh: bool = False) -> pd.DataFrame:
    """
    Запрашивает данные свечей с MOEX API с кэшированием и повторными попытками.
    
    Аргументы:
        ticker: Тикер инструмента
        interval: Интервал свечей в минутах
        period: Период данных ('1D' - 1 день, '2D' - 2 дня и т.д. или 'M1' - 1 минута и т.д.)
        end: Конечная дата (если None, используется текущая дата)
        record: Флаг для сохранения данных в файл
        force_refresh: Флаг для принудительного обновления кэша
        
    Возвращает:
        pandas.DataFrame с данными свечей
        
    Вызывает:
        APK.InvalidInputError: при неверном формате периода
        APK.DatabaseError: при пустом результате запроса
        APK.APIError: при проблемах с API
    """
    # Определение временного интервала
    start = None
    
    if re.fullmatch(r'\d+[A-Za-z]', period):  # Формат "ЧислоБуква" (например, "1D")
        days = int(re.match(r'(\d+)[A-Za-z]', period).group(1))
        buf = datetime.datetime.now() - datetime.timedelta(days=days)
        start = buf.strftime('%Y-%m-%d')
    elif re.fullmatch(r'[A-Za-z]\d+', period):  # Формат "БукваЧисло" (например, "M1")
        minutes = int(re.match(r'[A-Za-z](\d+)', period).group(1))
        buf = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
        start = buf.strftime('%Y-%m-%d')
    else:
        raise APK.InvalidInputError(f"Неверный формат периода: {period}. Ожидается формат '1D' или 'M1'.")
    
    # Проверка кэша
    cache_key = (ticker, interval, start, end)
    if not force_refresh and is_cache_valid(cache_key):
        print(f"Используется кэшированный результат для {ticker}")
        return REQUEST_CACHE[cache_key][1]
    
    # Запрос к API с повторными попытками
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            with requests.Session() as session:
                candles = apimoex.get_board_candles(session, ticker, interval, start, end)
                
                # Проверка на пустоту результата
                if not candles:
                    raise APK.DatabaseError(f"Пустой результат запроса для {ticker} с {start} по {end}")
                
                # Преобразование в DataFrame
                dataFrame = pd.DataFrame(candles)
                
                # Кэширование результата
                REQUEST_CACHE[cache_key] = (time.time(), dataFrame)
                
                # Сохранение в файл, если требуется
                if record:
                    timestamp = datetime.datetime.now().strftime("[%H%M%S]")
                    filename = f"{ticker}_{start}_{period}_{interval}_{timestamp}.json"
                    filepath = os.path.join(DATA_DIR, filename)
                    dataFrame.to_json(filepath)
                    print(f"Данные сохранены в {filepath}")
                
                return dataFrame
                
        except requests.exceptions.HTTPError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                raise APK.APIError(f"HTTP ошибка при запросе к MOEX API: {e}")
            print(f"HTTP ошибка ({retry_count}/{MAX_RETRIES}), повторная попытка: {e}")
            time.sleep(1)  # Пауза перед повторной попыткой
            
        except requests.exceptions.ConnectionError as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                raise APK.APIError(f"Ошибка соединения с MOEX API: {e}")
            print(f"Ошибка соединения ({retry_count}/{MAX_RETRIES}), повторная попытка: {e}")
            time.sleep(2)  # Более длинная пауза при ошибке соединения
            
        except requests.exceptions.Timeout as e:
            retry_count += 1
            if retry_count >= MAX_RETRIES:
                raise APK.APIError(f"Тайм-аут запроса к MOEX API: {e}")
            print(f"Тайм-аут ({retry_count}/{MAX_RETRIES}), повторная попытка: {e}")
            time.sleep(2)
            
        except requests.exceptions.InvalidURL as e:
            raise APK.APIError(f"Некорректный URL для MOEX API: {e}")


def get_available_tickers():
    """
    Получает список доступных тикеров с MOEX.
    
    Возвращает:
        list: Список доступных тикеров
    """
    try:
        with requests.Session() as session:
            # Можно настроить под нужный рынок/режим торгов
            securities = apimoex.get_board_securities(session, 'TQBR')
            return [security['SECID'] for security in securities]
    except Exception as e:
        raise APK.APIError(f"Ошибка при получении списка тикеров: {e}")


def save_data_with_metadata(dataFrame: pd.DataFrame, 
                          ticker: str, 
                          interval: int, 
                          period: str) -> str:
    """
    Сохраняет данные с метаданными для улучшенного управления файлами.
    
    Аргументы:
        dataFrame: pandas.DataFrame с данными для сохранения
        ticker: Тикер инструмента
        interval: Интервал свечей
        period: Период данных
        
    Возвращает:
        str: Путь к сохраненному файлу
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    metadata = {
        "ticker": ticker,
        "interval": interval,
        "period": period,
        "created_at": timestamp,
        "rows_count": len(dataFrame)
    }
    
    # Создаем имя файла
    filename = f"{ticker}_{timestamp}_{period}_{interval}.json"
    filepath = os.path.join(DATA_DIR, filename)
    
    # Сохраняем данные и метаданные
    with open(filepath, 'w', encoding='utf-8') as f:
        # Преобразуем DataFrame в словарь для добавления метаданных
        data_dict = dataFrame.to_dict(orient='records')
        output = {
            "metadata": metadata,
            "data": data_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Данные сохранены в {filepath}")
    return filepath


if __name__ == "__main__":
    try:
        # Пример использования функций модуля
        print("Доступные тикеры (первые 10):")
        tickers = get_available_tickers()
        print(tickers[:10])
        
        print("\nЗапрос данных для SBER:")
        fetchedData = ask_moex(ticker="SBER", interval=1, period="9D", record=True)
        print(f"Получено строк: {len(fetchedData)}")
        print(fetchedData.head())
        
        # Пример сохранения с метаданными
        save_data_with_metadata(fetchedData, "SBER", 1, "9D")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")