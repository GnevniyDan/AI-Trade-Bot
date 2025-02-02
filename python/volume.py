import json
import os
import pandas as pd
from datetime import datetime
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def load_data(filename):
    """Загружает данные из JSON-файла в DataFrame."""
    try:
        full_path = os.path.join(DATA_DIR, filename)
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Для структуры данных FEES
        if isinstance(data, dict) and all(key in data for key in ['begin', 'close', 'volume']):
            df = pd.DataFrame({
                'date': list(data['begin'].values()),
                'close': list(data['close'].values()),
                'volume': list(data['volume'].values())
            })
        else:
            df = pd.DataFrame(data)
        
        # Проверка наличия необходимых столбцов
        required_columns = ['close', 'volume', 'date']
        if not all(col in df.columns for col in required_columns):
            raise APK.InvalidInputError("Отсутствуют обязательные столбцы: 'close', 'volume' или 'date'")
        
        df['close'] = pd.to_numeric(df['close'])
        df['volume'] = pd.to_numeric(df['volume'])
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except FileNotFoundError:
        raise APK.DatabaseError("Файл данных не найден.")
    except json.JSONDecodeError:
        raise APK.InvalidInputError("Ошибка при чтении JSON-файла.")
    except ValueError as e:
        raise APK.InvalidInputError(f"Ошибка преобразования данных: {str(e)}")

def find_support_resistance(df, lookback=20):
    """Находит уровни поддержки и сопротивления используя pandas."""
    try:
        recent_prices = df['close'].tail(lookback)
        support = recent_prices.min()
        resistance = recent_prices.max()
        
        return APK.todaySupRes(
            pivot=(support + resistance) / 2,
            resistance_1=resistance,
            resistance_2=resistance * 1.05,
            resistance_3=resistance * 1.1,
            support_1=support,
            support_2=support * 0.95,
            support_3=support * 0.9
        )
    except Exception as e:
        raise APK.InvalidInputError(f"Ошибка при расчете уровней: {str(e)}")

def volume_analysis(df, support, resistance):
    """
    Анализирует объемы и цены, возвращает DataFrame с сигналами.
    
    Сигналы:
    2  = Сильный сигнал на покупку (прорыв сопротивления с высоким объемом)
    1  = Слабый сигнал на покупку (рост цены при высоком объеме)
    0  = Нет сигнала
    -1 = Слабый сигнал на продажу (падение цены при высоком объеме)
    -2 = Сильный сигнал на продажу (прорыв поддержки с высоким объемом)
    """
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['Volume_Signal'] = 0
    
    # Сильный сигнал на покупку
    df.loc[(df['close'] > resistance) & 
           (df['volume'] > df['prev_volume']), 'Volume_Signal'] = 2
    
    # Сильный сигнал на продажу
    df.loc[(df['close'] < support) & 
           (df['volume'] > df['prev_volume']), 'Volume_Signal'] = -2
    
    # Слабый сигнал на покупку
    df.loc[(df['close'] > df['prev_close']) & 
           (df['volume'] > df['prev_volume']) & 
           (df['Volume_Signal'] == 0), 'Volume_Signal'] = 1
    
    # Слабый сигнал на продажу
    df.loc[(df['close'] < df['prev_close']) & 
           (df['volume'] > df['prev_volume']) & 
           (df['Volume_Signal'] == 0), 'Volume_Signal'] = -1
    
    return df

def get_volume_summary(data: pd.DataFrame) -> dict:
    """
    Создает сводку по текущим значениям объемов и сигналам
    
    Returns:
        dict с текущим состоянием индикатора
    """
    try:
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Определяем тренд объема
        if latest['volume'] > prev['volume']:
            volume_trend = "Растущий"
        elif latest['volume'] < prev['volume']:
            volume_trend = "Падающий"
        else:
            volume_trend = "Боковой"
            
        # Расшифровка сигнала
        signal_desc = {
            2: "Сильный сигнал на покупку",
            1: "Слабый сигнал на покупку",
            0: "Нет сигнала",
            -1: "Слабый сигнал на продажу",
            -2: "Сильный сигнал на продажу"
        }
            
        return {
            'volume_trend': volume_trend,
            'current_volume': latest['volume'],
            'prev_volume': prev['volume'],
            'volume_change': (latest['volume'] - prev['volume']) / prev['volume'] * 100,
            'signal': latest['Volume_Signal'],
            'signal_description': signal_desc[latest['Volume_Signal']]
        }
        
    except Exception as e:
        raise APK.ApplicationError(f"Error creating volume summary: {str(e)}")

def main(filename):
    try:
        # Чтение данных
        df = load_data(filename)
        
        # Нахождение уровней
        sup_res = find_support_resistance(df)
        print("\nУровни поддержки и сопротивления:")
        print(sup_res)
        
        # Анализ объемов
        result = volume_analysis(df, sup_res.support_1, sup_res.resistance_1)
        
        # Получение сводки
        summary = get_volume_summary(result)
        
        # Вывод результатов
        print("\nПоследние 10 значений объемов и сигналов:")
        print(result[['close', 'volume', 'Volume_Signal']].tail(10))
        
        print("\nТекущее состояние:")
        print(f"Тренд объема: {summary['volume_trend']}")
        print(f"Текущий объем: {summary['current_volume']:,.0f}")
        print(f"Изменение объема: {summary['volume_change']:.2f}%")
        print(f"Сигнал ({summary['signal']}): {summary['signal_description']}")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")

if __name__ == "__main__":
    filename = "FEES_2024-11-10_3D_[183522].json"
    main(filename)