import requests
import apimoex
import pandas
import datetime
import os 


# Путь к директории для сохранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def ask_moex(ticker: str = "FEES", interval: int = 10, period: str = "1D") -> pandas.DataFrame:
    with requests.Session() as session:

        #установка анализируемого интервала
        match list(period):
            #_D сколько-то дней
            case [param, "D"] if param.isnumeric(): 
                buf = datetime.datetime.now() - datetime.timedelta(days=int(param))
                start = buf.strftime('%Y-%m-%d')
                end = ""

            #M_ сколько-то минут
            case ["M", param] if param.isnumeric(): 
                buf = datetime.datetime.now() - datetime.timedelta(minutes=int(param))
                start = buf.strftime('%Y-%m-%d')
                end = ""

        #список словарей свечек
        candles = apimoex.get_board_candles(session, ticker, interval, start, end)
        if not candles : return

        #перегон в датафрейм
        dataFrame = pandas.DataFrame(candles)
        dataFrame.to_json("storage/{}_from_{}.json".format(ticker, start))
        print("Created files:\nstorage/{}_from_{}.json".format(ticker, start))
        return dataFrame


if __name__ == "__main__":
    fetchedData = ask_moex(ticker = "KMAZ", period = "3D")
    print(fetchedData)