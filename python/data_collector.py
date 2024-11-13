import requests
import apimoex
import pandas
import datetime
import os 
import re as regular
import _AppProjectKit as APK


"""class ApplicationError(Exception):
    pass

class DatabaseError(ApplicationError):
    pass

class InvalidInputError(ApplicationError):
    def __init__(self, message="Некорректный ввод данных"):
        self.message = message
        super().__init__(self.message)"""
        
        
# Путь к директории для сохранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def ask_moex(ticker: str = "FEES", interval: int = 10, period: str = "1D", end: str = None) -> pandas.DataFrame:
    with requests.Session() as session:

        #установка анализируемого интервала
        match period:
            #_D сколько-то дней
            case period if regular.fullmatch(r'\d[A-Za-z]', period): 
                buf = datetime.datetime.now() - datetime.timedelta(days=int(period[0]))
                start = buf.strftime('%Y-%m-%d')

            #M_ сколько-то минут
            case period if regular.fullmatch(r'[A-Za-z]\d', period): 
                buf = datetime.datetime.now() - datetime.timedelta(minutes=int(period[1]))
                start = buf.strftime('%Y-%m-%d')

            #Ошибка ввода
            case _:
                raise APK.InvalidInputError("wrong period")
            
        #список словарей свечек
        try:
            candles = apimoex.get_board_candles(session, ticker, interval, start, end)

        except requests.exceptions.HTTPError as re:
            print(f"HTTP ошибка: {re}")
            
        except requests.exceptions.ConnectionError as ce:
            print(f"Ошибка соединения: {ce}")

        except requests.exceptions.Timeout as t:
            print(f"Ошибка тайм-аута: {t}")

        except requests.exceptions.InvalidURL as url:
            print(f"Некорректный URL: {url}")

        #проверка на пустоту
        if not candles:
            raise  APK.DatabaseError("dataframe is empty")

        #перегон в датафрейм
        dataFrame = pandas.DataFrame(candles)
        buf = buf.strftime("[%H%M%S]")
        dataFrame.to_json("storage/{}_{}_{}_{}.json".format(ticker, start, period, buf))
        return dataFrame


if __name__ == "__main__":
    fetchedData = ask_moex(ticker = "MOEX", period = "1D")
    print(fetchedData)