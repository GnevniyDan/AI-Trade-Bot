import pandas
import numpy
import _AppProjectKit as APK
import os
from rsi_call import current_rsi_call

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def calculate_stochastic_rsi(dataFrame: pandas.DataFrame, period: int = 14, 
                           smooth_k: int = 3, smooth_d: int = 3) -> pandas.DataFrame:
    """
    Рассчитывает Стохастический RSI (StochRSI).
    
    Args:
        dataFrame: pandas DataFrame с данными
        period: период для расчёта RSI и Стохастического осциллятора
        smooth_k: период сглаживания для %K линии
        smooth_d: период сглаживания для %D линии
    
    Returns:
        pandas DataFrame с добавленными колонками:
        - RSI: значения RSI
        - StochRSI_K: быстрая линия StochRSI
        - StochRSI_D: медленная линия StochRSI
        - StochRSI_Signal: сигналы (-1, 0, 1)
    """
    try:
        # Копируем DataFrame
        df = dataFrame.copy()
        
        # Получаем значения RSI
        df['RSI'] = current_rsi_call(df, period)
        
        # Рассчитываем Стохастический RSI
        rsi_series = df['RSI']
        lowest_low = rsi_series.rolling(window=period).min()
        highest_high = rsi_series.rolling(window=period).max()
        
        # Формула Стохастического RSI
        # %K = (RSI - MIN(RSI, period)) / (MAX(RSI, period) - MIN(RSI, period)) * 100
        df['StochRSI_K'] = 100 * (rsi_series - lowest_low) / (highest_high - lowest_low)
        
        # Сглаживаем %K
        df['StochRSI_K'] = df['StochRSI_K'].rolling(window=smooth_k).mean()
        
        # Рассчитываем %D (сглаженная версия %K)
        df['StochRSI_D'] = df['StochRSI_K'].rolling(window=smooth_d).mean()
        
        # Генерируем сигналы
        df['StochRSI_Signal'] = 0
        
        # Сигнал на покупку: StochRSI_K пересекает StochRSI_D снизу вверх и оба < 20
        df.loc[(df['StochRSI_K'] > df['StochRSI_D']) & 
               (df['StochRSI_K'].shift(1) <= df['StochRSI_D'].shift(1)) &
               (df['StochRSI_K'] < 20), 'StochRSI_Signal'] = 1
               
        # Сигнал на продажу: StochRSI_K пересекает StochRSI_D сверху вниз и оба > 80
        df.loc[(df['StochRSI_K'] < df['StochRSI_D']) & 
               (df['StochRSI_K'].shift(1) >= df['StochRSI_D'].shift(1)) &
               (df['StochRSI_K'] > 80), 'StochRSI_Signal'] = -1
        
        return df
        
    except Exception as e:
        raise APK.ApplicationError(f"Error calculating Stochastic RSI: {str(e)}")

def get_stoch_rsi_summary(data: pandas.DataFrame) -> dict:
    """
    Создает сводку по текущим значениям Stochastic RSI
    
    Args:
        data: DataFrame с рассчитанными показателями StochRSI
        
    Returns:
        dict с текущим состоянием индикатора
    """
    try:
        latest = data.iloc[-1]
        
        # Определяем состояние
        if latest['StochRSI_K'] > 80:
            condition = "Перекуплен"
        elif latest['StochRSI_K'] < 20:
            condition = "Перепродан"
        else:
            condition = "Нейтральный"
            
        # Определяем тренд
        if latest['StochRSI_K'] > latest['StochRSI_D']:
            trend = "Восходящий"
        elif latest['StochRSI_K'] < latest['StochRSI_D']:
            trend = "Нисходящий"
        else:
            trend = "Боковой"
            
        return {
            'condition': condition,
            'trend': trend,
            'rsi': latest['RSI'],
            'stoch_k': latest['StochRSI_K'],
            'stoch_d': latest['StochRSI_D'],
            'signal': latest['StochRSI_Signal']
        }
        
    except Exception as e:
        raise APK.ApplicationError(f"Error creating Stochastic RSI summary: {str(e)}")

if __name__ == "__main__":
    try:
        # Чтение данных из файла
        data = pandas.read_json("storage/FEES_2024-11-10_3D_[183522].json")
        
        # Расчет Stochastic RSI
        result = calculate_stochastic_rsi(data)
        
        # Получение сводки
        summary = get_stoch_rsi_summary(result)
        
        # Вывод результатов
        print("\nПоследние 10 значений Stochastic RSI:")
        print(result[['RSI', 'StochRSI_K', 'StochRSI_D', 'StochRSI_Signal']].tail(10))
        
        print("\nТекущее состояние:")
        print(f"Состояние: {summary['condition']}")
        print(f"Тренд: {summary['trend']}")
        print(f"RSI: {summary['rsi']:.2f}")
        print(f"Стохастик %K: {summary['stoch_k']:.2f}")
        print(f"Стохастик %D: {summary['stoch_d']:.2f}")
        print(f"Сигнал: {summary['signal']}")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")