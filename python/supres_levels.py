
import pandas as pd
import numpy as np
import datetime
import os
import re
import _AppProjectKit as APK
from typing import List, Dict, Tuple, Union, Optional

# Директория для хранения данных
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)  # Создать директорию, если она отсутствует

# Константы для расчета уровней
DEFAULT_LOOKBACK = 30  # Стандартное количество дней для анализа
MIN_PIVOT_STRENGTH = 3  # Минимальное количество точек для подтверждения уровня
PRICE_PROXIMITY_PCT = 0.01  # Процент близости для определения касания уровня


def identify_pivot_points(data: pd.DataFrame) -> pd.DataFrame:
    """
    Идентифицирует точки разворота в ценовом ряду.
    
    Аргументы:
        data: DataFrame с ценовыми данными
        
    Возвращает:
        DataFrame с отмеченными точками разворота
    """
    if len(data) < 5:
        raise APK.InvalidInputError("Недостаточно данных для определения точек разворота (нужно минимум 5 точек)")
    
    df = data.copy()
    
    # Добавляем столбцы для локальных максимумов и минимумов
    df['local_max'] = False
    df['local_min'] = False
    
    # Находим локальные максимумы (цена выше, чем у соседей)
    for i in range(2, len(df) - 2):
        # Максимум
        if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
            df['high'].iloc[i] > df['high'].iloc[i-2] and
            df['high'].iloc[i] > df['high'].iloc[i+1] and
            df['high'].iloc[i] > df['high'].iloc[i+2]):
            df['local_max'].iloc[i] = True
        
        # Минимум
        if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
            df['low'].iloc[i] < df['low'].iloc[i-2] and
            df['low'].iloc[i] < df['low'].iloc[i+1] and
            df['low'].iloc[i] < df['low'].iloc[i+2]):
            df['local_min'].iloc[i] = True
    
    return df


def calculate_historical_levels(data: pd.DataFrame, lookback: int = DEFAULT_LOOKBACK) -> Dict[str, List[Tuple[float, int]]]:
    """
    Рассчитывает исторические уровни поддержки и сопротивления.
    
    Аргументы:
        data: DataFrame с ценовыми данными
        lookback: Количество дней для анализа
        
    Возвращает:
        Словарь с уровнями и их силой
    """
    if 'local_max' not in data.columns or 'local_min' not in data.columns:
        data = identify_pivot_points(data)
    
    # Получаем подмножество данных для анализа
    recent_data = data.iloc[-lookback:] if len(data) > lookback else data
    
    # Список локальных максимумов и минимумов
    highs = recent_data[recent_data['local_max']]['high'].tolist()
    lows = recent_data[recent_data['local_min']]['low'].tolist()
    
    # Кластеризация похожих уровней
    resistance_clusters = cluster_price_levels(highs)
    support_clusters = cluster_price_levels(lows)
    
    # Оцениваем силу каждого уровня по количеству касаний
    resistance_levels = score_levels(recent_data, resistance_clusters, 'resistance')
    support_levels = score_levels(recent_data, support_clusters, 'support')
    
    return {
        'resistance': resistance_levels,
        'support': support_levels
    }


def cluster_price_levels(price_points: List[float], proximity_pct: float = PRICE_PROXIMITY_PCT) -> List[float]:
    """
    Группирует близкие ценовые уровни в кластеры.
    
    Аргументы:
        price_points: Список цен для кластеризации
        proximity_pct: Процент близости для группировки
        
    Возвращает:
        Список кластеризованных уровней
    """
    if not price_points:
        return []
    
    # Сортируем уровни
    sorted_prices = sorted(price_points)
    
    clusters = []
    current_cluster = [sorted_prices[0]]
    
    for price in sorted_prices[1:]:
        # Если цена находится в пределах заданного процента от среднего кластера
        cluster_avg = sum(current_cluster) / len(current_cluster)
        if abs(price - cluster_avg) / cluster_avg <= proximity_pct:
            current_cluster.append(price)
        else:
            # Добавляем средний уровень текущего кластера и начинаем новый кластер
            clusters.append(sum(current_cluster) / len(current_cluster))
            current_cluster = [price]
    
    # Добавляем последний кластер
    if current_cluster:
        clusters.append(sum(current_cluster) / len(current_cluster))
    
    return clusters


def score_levels(data: pd.DataFrame, levels: List[float], level_type: str) -> List[Tuple[float, int]]:
    """
    Оценивает силу уровней поддержки/сопротивления.
    
    Аргументы:
        data: DataFrame с ценовыми данными
        levels: Список уровней для оценки
        level_type: Тип уровня ('support' или 'resistance')
        
    Возвращает:
        Список кортежей (уровень, сила)
    """
    scored_levels = []
    
    for level in levels:
        touches = 0
        
        # Для уровней сопротивления ищем касания сверху
        if level_type == 'resistance':
            for i in range(len(data)):
                if (data['high'].iloc[i] >= level * 0.997 and 
                    data['high'].iloc[i] <= level * 1.003):
                    touches += 1
        
        # Для уровней поддержки ищем касания снизу
        else:  # support
            for i in range(len(data)):
                if (data['low'].iloc[i] <= level * 1.003 and 
                    data['low'].iloc[i] >= level * 0.997):
                    touches += 1
        
        # Добавляем только сильные уровни
        if touches >= MIN_PIVOT_STRENGTH:
            scored_levels.append((level, touches))
    
    # Сортируем уровни по силе (по убыванию)
    return sorted(scored_levels, key=lambda x: x[1], reverse=True)


def today_levels(dataFrame: pd.DataFrame) -> APK.todaySupRes:
    """
    Рассчитывает дневные уровни поддержки и сопротивления для переданного DataFrame.
    Использует улучшенный алгоритм с учетом исторических уровней.

    Аргументы:
    dataFrame -- pandas DataFrame с свечами
    
    Возвращает:
    Объект класса todaySupRes
    """
    #фильтрация фрейма по последним свечам
    try:
        now = pd.to_datetime(dataFrame.iloc[-1]["begin"]) - pd.Timedelta(days=1)
        now_data = dataFrame.loc[dataFrame['begin'] >= now.strftime("%Y-%m-%d")]
    except (KeyError, ValueError):
        # Если не удается распарсить дату, берем последние 24 бара
        now_data = dataFrame.tail(24)

    #проверка на пустоту
    if now_data.empty: 
        raise APK.DatabaseError(f"dataframe is empty")

    #параметры
    high = now_data['high'].max()
    low = now_data['low'].min()
    close = now_data['close'].iloc[-1]
    
    # Рассчитываем уровни на основе формулы pivot point
    pivot = (high + low + close) / 3
    
    # Классический расчет уровней
    resistance_1 = 2 * pivot - low
    resistance_2 = pivot + (high - low)
    resistance_3 = pivot + 2 * (high - low)
    support_1 = 2 * pivot - high
    support_2 = pivot - (high - low)
    support_3 = pivot - 2 * (high - low)
    
    # Дополнительно находим исторические уровни
    try:
        historical_levels = calculate_historical_levels(dataFrame)
        
        # Корректируем основные уровни с учетом исторических данных
        if historical_levels['resistance']:
            # Ищем ближайший исторический уровень сопротивления выше текущей цены
            for level, strength in historical_levels['resistance']:
                if level > close:
                    # Чем сильнее исторический уровень, тем больше влияние
                    weight = min(0.3, strength * 0.05)  # максимум 30% влияния
                    resistance_1 = resistance_1 * (1 - weight) + level * weight
                    break
        
        if historical_levels['support']:
            # Ищем ближайший исторический уровень поддержки ниже текущей цены
            for level, strength in historical_levels['support']:
                if level < close:
                    # Чем сильнее исторический уровень, тем больше влияние
                    weight = min(0.3, strength * 0.05)  # максимум 30% влияния
                    support_1 = support_1 * (1 - weight) + level * weight
                    break
    except Exception as e:
        # В случае ошибки при расчете исторических уровней просто продолжаем с классическими
        print(f"Ошибка при расчете исторических уровней: {e}")
    
    # Создаем объект с уровнями
    levels_class = APK.todaySupRes(
        pivot=round(pivot, 2),
        resistance_1=round(resistance_1, 2),
        resistance_2=round(resistance_2, 2),
        resistance_3=round(resistance_3, 2),
        support_1=round(support_1, 2),
        support_2=round(support_2, 2),
        support_3=round(support_3, 2)
    )
    
    return levels_class


def find_key_levels(dataFrame: pd.DataFrame, n_levels: int = 5) -> Dict[str, List[float]]:
    """
    Находит ключевые уровни поддержки и сопротивления на основе кластерного анализа.
    
    Аргументы:
        dataFrame: DataFrame с ценовыми данными
        n_levels: Количество уровней для возврата
        
    Возвращает:
        Словарь с ключевыми уровнями поддержки и сопротивления
    """
    # Проверка наличия необходимых столбцов
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in dataFrame.columns]
    if missing_columns:
        raise APK.InvalidInputError(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
    
    # Находим точки разворота
    pivot_data = identify_pivot_points(dataFrame)
    
    # Рассчитываем исторические уровни
    historical_levels = calculate_historical_levels(pivot_data)
    
    # Берем топ N уровней
    top_resistance = historical_levels['resistance'][:n_levels]
    top_support = historical_levels['support'][:n_levels]
    
    # Преобразуем в списки только уровней
    resistance_levels = [level for level, _ in top_resistance]
    support_levels = [level for level, _ in top_support]
    
    # Добавляем информацию о текущей цене для контекста
    current_price = dataFrame['close'].iloc[-1]
    
    return {
        'resistance': resistance_levels,
        'support': support_levels,
        'current_price': current_price
    }


def evaluate_level_strength(dataFrame: pd.DataFrame, price_level: float, level_type: str) -> int:
    """
    Оценивает силу конкретного ценового уровня.
    
    Аргументы:
        dataFrame: DataFrame с ценовыми данными
        price_level: Ценовой уровень для оценки
        level_type: Тип уровня ('support' или 'resistance')
        
    Возвращает:
        Оценка силы уровня (0-10)
    """
    # Определяем границы для проверки касаний (±0.5%)
    proximity = price_level * 0.005
    
    touches = 0
    rejections = 0
    
    for i in range(len(dataFrame)):
        # Для сопротивления
        if level_type == 'resistance':
            # Касание - высокая цена рядом с уровнем
            if abs(dataFrame['high'].iloc[i] - price_level) <= proximity:
                touches += 1
                
                # Отскок - после касания цена пошла вниз
                if i < len(dataFrame) - 1 and dataFrame['close'].iloc[i+1] < dataFrame['close'].iloc[i]:
                    rejections += 1
        
        # Для поддержки
        else:  # 'support'
            # Касание - низкая цена рядом с уровнем
            if abs(dataFrame['low'].iloc[i] - price_level) <= proximity:
                touches += 1
                
                # Отскок - после касания цена пошла вверх
                if i < len(dataFrame) - 1 and dataFrame['close'].iloc[i+1] > dataFrame['close'].iloc[i]:
                    rejections += 1
    
    if touches == 0:
        return 0
    
    # Рассчитываем силу на основе количества касаний и процента отскоков
    rejection_rate = rejections / touches
    strength = min(10, int((touches * 0.5 + rejection_rate * 5)))
    
    return strength


if __name__ == "__main__":
    # Чтение данных из файла
    try:
        data_file = os.path.join(DATA_DIR, "MOEX_2024-11-12_1D_[191120].json")
        if os.path.exists(data_file):
            data = pd.read_json(data_file)
        else:
            # Создаем тестовые данные для демонстрации
            print("Файл данных не найден. Создаем тестовые данные...")
            
            # Генерируем тестовые данные
            dates = pd.date_range(start='2023-01-01', periods=100)
            np.random.seed(42)
            
            price = 100.0
            prices = []
            for _ in range(100):
                change = np.random.normal(0, 2)
                price += change
                prices.append(price)
            
            data = pd.DataFrame({
                'begin': dates.strftime('%Y-%m-%d %H:%M:%S'),
                'close': prices,
                'high': [p + np.random.uniform(0, 1) for p in prices],
                'low': [p - np.random.uniform(0, 1) for p in prices],
                'open': [p + np.random.uniform(-1, 1) for p in prices],
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Сохраняем тестовые данные
            test_file = os.path.join(DATA_DIR, "test_supres_data.json")
            data.to_json(test_file)
            print(f"Тестовые данные сохранены в {test_file}")
        
        # Проверка наличия необходимого столбца 'begin'
        if 'begin' not in data.columns:
            raise ValueError("Data does not contain 'begin' column.")
        
        # Расчет уровней
        levels = today_levels(data)
        print(levels)
        
        # Дополнительно находим ключевые исторические уровни
        key_levels = find_key_levels(data)
        
        print("\nКлючевые исторические уровни:")
        print("Сопротивление:", ', '.join(f"{level:.2f}" for level in key_levels['resistance']))
        print("Поддержка:", ', '.join(f"{level:.2f}" for level in key_levels['support']))
        print(f"Текущая цена: {key_levels['current_price']:.2f}")
        
    except APK.ApplicationError as e:
        print(f"Ошибка приложения: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")