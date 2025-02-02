import pandas
import numpy
import _AppProjectKit as APK
import os

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует

def current_ma_analysis(dataFrame: pandas.DataFrame, short_period: int = 9, long_period: int = 21) -> pandas.DataFrame:
    """
    Добавляет колонки с индикаторами скользящих средних в DataFrame.
    
    Args:
        dataFrame: pandas DataFrame с данными
        short_period: период короткой скользящей средней (по умолчанию 9)
        long_period: период длинной скользящей средней (по умолчанию 21)
    
    Returns:
        pandas DataFrame с добавленными колонками:
        - SMA: простая скользящая средняя
        - EMA: экспоненциальная скользящая средняя
        - Volatility: волатильность
        - MA_Signal: сигнал от пересечения (-1, 0, 1)
    """
    # Проверка наличия необходимого столбца
    if 'close' not in dataFrame.columns:
        raise APK.InvalidInputError("Data does not contain 'close' column.")
    
    # Проверка достаточной длины данных
    if len(dataFrame) < long_period:
        raise APK.DatabaseError(f"Not enough data points. Required: {long_period}, Got: {len(dataFrame)}")
    
    # Создаем копию DataFrame
    df = dataFrame.copy()
    
    try:
        # Добавляем SMA
        df['SMA'] = df['close'].rolling(window=long_period).mean()
        
        # Добавляем EMA
        df['EMA'] = df['close'].ewm(span=short_period, adjust=False).mean()
        
        # Добавляем волатильность (на основе дневных изменений)
        # Используем краткосрочный период для более чувствительного измерения
        df['Volatility'] = df['close'].pct_change().rolling(
            window=short_period).std() * numpy.sqrt(short_period)
        
        # Добавляем сигналы от пересечения MA
        df['MA_Signal'] = 0  # По умолчанию нет сигнала
        
        # Рассчитываем короткую и длинную MA для сигналов
        short_ma = df['close'].ewm(span=short_period, adjust=False).mean()
        long_ma = df['close'].ewm(span=long_period, adjust=False).mean()
        
        # Сигнал на покупку (1): короткая MA пересекает длинную MA снизу вверх
        df.loc[(short_ma > long_ma) & 
               (short_ma.shift(1) <= long_ma.shift(1)), 'MA_Signal'] = 1
        
        # Сигнал на продажу (-1): короткая MA пересекает длинную MA сверху вниз
        df.loc[(short_ma < long_ma) & 
               (short_ma.shift(1) >= long_ma.shift(1)), 'MA_Signal'] = -1
        
        return df
        
    except Exception as e:
        raise APK.ApplicationError(f"Error calculating MA indicators: {str(e)}")

def get_ma_summary(dataFrame: pandas.DataFrame) -> dict:
    """
    Возвращает сводную информацию по индикаторам MA.
    
    Args:
        dataFrame: pandas DataFrame с рассчитанными индикаторами MA
    
    Returns:
        dict с краткой информацией о текущем состоянии индикаторов
    """
    try:
        latest = dataFrame.iloc[-1]
        
        # Определяем тренд
        if latest['EMA'] > latest['SMA']:
            trend = "Восходящий"
        elif latest['EMA'] < latest['SMA']:
            trend = "Нисходящий"
        else:
            trend = "Боковой"
            
        # Оцениваем волатильность
        avg_volatility = dataFrame['Volatility'].mean()
        current_volatility = latest['Volatility']
        
        if current_volatility > avg_volatility * 1.5:
            volatility_status = "Высокая"
        elif current_volatility < avg_volatility * 0.5:
            volatility_status = "Низкая"
        else:
            volatility_status = "Средняя"
            
        return {
            'trend': trend,
            'volatility_status': volatility_status,
            'current_volatility': current_volatility,
            'signal': latest['MA_Signal'],
            'sma': latest['SMA'],
            'ema': latest['EMA'],
            'close': latest['close']
        }
        
    except Exception as e:
        raise APK.ApplicationError(f"Error creating MA summary: {str(e)}")

if __name__ == "__main__":
    try:
        # Чтение данных из файла
        data = pandas.read_json("storage/FEES_2024-11-10_3D_[183522].json")
        
        # Добавление индикаторов MA
        data_with_ma = current_ma_analysis(data)
        
        # Получение сводки
        summary = get_ma_summary(data_with_ma)
        
        # Вывод результатов
        print("\nПоследние 10 значений индикаторов MA:")
        print(data_with_ma[['close', 'SMA', 'EMA', 'Volatility', 'MA_Signal']].tail(10))
        
        print("\nСводная информация:")
        print(f"Тренд: {summary['trend']}")
        print(f"Волатильность: {summary['volatility_status']} ({summary['current_volatility']:.2%})")
        print(f"Сигнал: {summary['signal']}")
        print(f"Цена закрытия: {summary['close']:.2f}")
        print(f"SMA: {summary['sma']:.2f}")
        print(f"EMA: {summary['ema']:.2f}")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")