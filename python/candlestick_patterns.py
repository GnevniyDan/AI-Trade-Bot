
import pandas as pd
import numpy as np
import _AppProjectKit as APK
import os
from typing import Dict, Union, List

# Директория для хранения данных
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Константы для распознавания паттернов
DOJI_THRESHOLD = 0.05   # Максимальное отношение тела к диапазону для дожи
TREND_WINDOW = 5        # Окно для определения тренда
MIN_CONFIDENCE = 0.6    # Минимальная уверенность для сигнала

# Паттерны и их значения (положительные - бычьи, отрицательные - медвежьи)
PATTERNS = {
    "Hammer": 1.0,           # Молот (бычий)
    "HangingMan": -1.0,      # Повешенный (медвежий)
    "Engulfing_Bullish": 1.0, # Бычье поглощение
    "Engulfing_Bearish": -1.0, # Медвежье поглощение
    "Doji": 0.0,             # Дожи (нейтральный)
    "ShootingStar": -1.0     # Падающая звезда (медвежий)
}


def calculate_candle_properties(row):
    """Расчет характеристик свечи"""
    body = abs(row['close'] - row['open'])
    total_range = row['high'] - row['low']
    
    # Избегаем деления на ноль
    body_ratio = body / total_range if total_range != 0 else 0
    
    lower_shadow = min(row['open'], row['close']) - row['low']
    upper_shadow = row['high'] - max(row['open'], row['close'])
    
    # Расчет дополнительных характеристик
    lower_shadow_ratio = lower_shadow / total_range if total_range != 0 else 0
    upper_shadow_ratio = upper_shadow / total_range if total_range != 0 else 0
    
    # Цвет свечи (1 - бычья, -1 - медвежья, 0 - нейтральная)
    candle_color = 1 if row['close'] > row['open'] else -1 if row['close'] < row['open'] else 0
    
    return {
        'body': body,
        'body_ratio': body_ratio,
        'lower_shadow': lower_shadow,
        'upper_shadow': upper_shadow,
        'lower_shadow_ratio': lower_shadow_ratio,
        'upper_shadow_ratio': upper_shadow_ratio,
        'total_range': total_range,
        'color': candle_color
    }


def is_trend(dataFrame, index, window=TREND_WINDOW, direction='up'):
    """Определение тренда в указанном направлении"""
    if index < window:
        return False
        
    # Получаем срез данных
    slice_data = dataFrame.iloc[index-window:index]
    
    # Линейная регрессия для определения направления тренда
    x = np.arange(window)
    y = slice_data['close'].values
    slope, _ = np.polyfit(x, y, 1)
    
    # Требуется восходящий тренд
    if direction == 'up':
        return slope > 0
    # Требуется нисходящий тренд
    else:
        return slope < 0


def is_hammer(row, dataFrame, index, window=TREND_WINDOW):
    """Определение молота"""
    props = calculate_candle_properties(row)
    
    # Молот: маленькое тело, длинная нижняя тень, очень короткая верхняя тень, в нисходящем тренде
    basic_conditions = (
        props['lower_shadow_ratio'] > 0.65 and  # Длинная нижняя тень
        props['upper_shadow_ratio'] < 0.1 and   # Минимальная верхняя тень
        props['body_ratio'] < 0.3 and           # Маленькое тело
        is_trend(dataFrame, index, window, 'down')  # Нисходящий тренд
    )
    
    if not basic_conditions:
        return 0.0
    
    # Расчет уверенности в сигнале
    confidence = 1.0
    
    # Уменьшаем уверенность, если объем ниже среднего
    avg_volume = dataFrame['volume'].rolling(window=20).mean().iloc[index]
    if row['volume'] < avg_volume:
        confidence *= 0.8
    
    return min(confidence, 1.0)  # Максимум 1.0


def is_hanging_man(row, dataFrame, index, window=TREND_WINDOW):
    """Определение повешенного"""
    props = calculate_candle_properties(row)
    
    # Повешенный: маленькое тело, длинная нижняя тень, очень короткая верхняя тень, в восходящем тренде
    basic_conditions = (
        props['lower_shadow_ratio'] > 0.65 and  # Длинная нижняя тень
        props['upper_shadow_ratio'] < 0.1 and   # Минимальная верхняя тень
        props['body_ratio'] < 0.3 and           # Маленькое тело
        is_trend(dataFrame, index, window, 'up')  # Восходящий тренд
    )
    
    if not basic_conditions:
        return 0.0
    
    # Расчет уверенности в сигнале
    confidence = 1.0
    
    # Уменьшаем уверенность, если объем ниже среднего
    avg_volume = dataFrame['volume'].rolling(window=20).mean().iloc[index]
    if row['volume'] < avg_volume:
        confidence *= 0.8
    
    return min(confidence, 1.0)  # Максимум 1.0


def is_shooting_star(row, dataFrame, index, window=TREND_WINDOW):
    """Определение падающей звезды"""
    props = calculate_candle_properties(row)
    
    # Падающая звезда: маленькое тело, очень короткая нижняя тень, длинная верхняя тень, в восходящем тренде
    basic_conditions = (
        props['upper_shadow_ratio'] > 0.65 and  # Длинная верхняя тень
        props['lower_shadow_ratio'] < 0.1 and   # Минимальная нижняя тень
        props['body_ratio'] < 0.3 and           # Маленькое тело
        is_trend(dataFrame, index, window, 'up')  # Восходящий тренд
    )
    
    if not basic_conditions:
        return 0.0
    
    # Расчет уверенности
    confidence = 1.0
    
    # Выше уверенность, если тело красное
    if props['color'] == -1:
        confidence *= 1.1
    
    return min(confidence, 1.0)


def is_engulfing(dataFrame, index):
    """Определение паттерна поглощения"""
    if index == 0:
        return False
    
    prev_row = dataFrame.iloc[index - 1]
    curr_row = dataFrame.iloc[index]
    
    prev_props = calculate_candle_properties(prev_row)
    curr_props = calculate_candle_properties(curr_row)
    
    # Тела должны быть значимыми
    if prev_props['body_ratio'] < 0.1 or curr_props['body_ratio'] < 0.1:
        return False
    
    # Бычье поглощение
    if (prev_props['color'] == -1 and            # Предыдущая красная
        curr_props['color'] == 1 and             # Текущая зеленая
        curr_row['open'] < prev_row['close'] and # Открытие ниже
        curr_row['close'] > prev_row['open']):   # Закрытие выше
        
        confidence = min(curr_row['volume'] / prev_row['volume'], 1.5) * 0.8
        return f'Engulfing_Bullish_{confidence:.2f}'
    
    # Медвежье поглощение
    if (prev_props['color'] == 1 and             # Предыдущая зеленая
        curr_props['color'] == -1 and            # Текущая красная
        curr_row['open'] > prev_row['close'] and # Открытие выше
        curr_row['close'] < prev_row['open']):   # Закрытие ниже
        
        confidence = min(curr_row['volume'] / prev_row['volume'], 1.5) * 0.8
        return f'Engulfing_Bearish_{confidence:.2f}'
    
    return False


def is_doji(row):
    """Определение свечи типа дожи"""
    props = calculate_candle_properties(row)
    
    # Дожи: крайне маленькое тело
    if props['body_ratio'] <= DOJI_THRESHOLD:
        # Классический дожи имеет примерно равные тени
        shadow_diff = abs(props['upper_shadow_ratio'] - props['lower_shadow_ratio'])
        
        if shadow_diff < 0.2:
            confidence = 1.0 - props['body_ratio'] / DOJI_THRESHOLD
            return min(confidence, 1.0)
    
    return 0.0


def current_candlestick_patterns(dataFrame: pd.DataFrame, window: int = TREND_WINDOW) -> pd.DataFrame:
    """
    Анализ свечных паттернов с оценкой уверенности в сигналах
    
    Аргументы:
        dataFrame: DataFrame с данными
        window: Размер окна для анализа тренда
        
    Возвращает:
        DataFrame с результатами анализа паттернов
    """
    # Создаем копию для избежания предупреждений
    df = dataFrame.copy()
    
    # Инициализация колонок для паттернов
    df['Hammer'] = 0.0
    df['HangingMan'] = 0.0
    df['Engulfing'] = False
    df['Doji'] = 0.0
    df['ShootingStar'] = 0.0
    
    # Расчет паттернов для каждой свечи
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Проверка молота
        df.loc[df.index[i], 'Hammer'] = is_hammer(row, df, i, window)
        
        # Проверка повешенного
        df.loc[df.index[i], 'HangingMan'] = is_hanging_man(row, df, i, window)
        
        # Проверка поглощения
        engulfing_result = is_engulfing(df, i)
        if engulfing_result:
            df.loc[df.index[i], 'Engulfing'] = engulfing_result
        
        # Проверка дожи
        df.loc[df.index[i], 'Doji'] = is_doji(row)
        
        # Проверка падающей звезды
        df.loc[df.index[i], 'ShootingStar'] = is_shooting_star(row, df, i, window)
    
    # Фильтрация слабых сигналов
    for col in ['Hammer', 'HangingMan', 'Doji', 'ShootingStar']:
        df.loc[df[col] < MIN_CONFIDENCE, col] = 0
    
    return df[['Hammer', 'HangingMan', 'Engulfing', 'Doji', 'ShootingStar']]


def get_pattern_signals(patterns: pd.DataFrame) -> Dict[str, Union[str, float, Dict]]:
    """
    Анализирует последние паттерны и возвращает агрегированные сигналы
    
    Аргументы:
        patterns: DataFrame с обнаруженными паттернами
        
    Возвращает:
        Словарь с агрегированными сигналами
    """
    if patterns.empty:
        return {"overall": "нейтральный", "score": 0}
    
    # Получаем последнюю строку паттернов
    last_patterns = patterns.iloc[-1]
    
    # Считаем общий сигнал
    bull_score = 0.0
    bear_score = 0.0
    active_patterns = []
    
    # Анализируем каждый паттерн
    for pattern, value in last_patterns.items():
        if value == 0 or value is False:
            continue
            
        # Извлекаем название паттерна и уверенность
        pattern_name = pattern
        confidence = value if isinstance(value, float) else 1.0
        
        # Обрабатываем строковые результаты (например, "Engulfing_Bullish_0.85")
        if isinstance(value, str) and "_" in value:
            parts = value.split("_")
            if len(parts) >= 3 and parts[-1].replace(".", "").isdigit():
                pattern_name = "_".join(parts[:-1])
                confidence = float(parts[-1])
        
        # Определяем бычий или медвежий паттерн
        pattern_score = 0
        for key, score in PATTERNS.items():
            if key == pattern_name or key in pattern_name:
                pattern_score = score * confidence
                break
        
        # Суммируем баллы
        if pattern_score > 0:
            bull_score += pattern_score
        elif pattern_score < 0:
            bear_score += abs(pattern_score)
        
        # Добавляем в список активных паттернов
        active_patterns.append({
            "name": pattern_name,
            "confidence": confidence,
            "type": "бычий" if pattern_score > 0 else "медвежий" if pattern_score < 0 else "нейтральный"
        })
    
    # Определяем общий сигнал
    overall = "нейтральный"
    if bull_score > bear_score * 1.5:
        overall = "сильный бычий"
    elif bull_score > bear_score:
        overall = "бычий"
    elif bear_score > bull_score * 1.5:
        overall = "сильный медвежий"
    elif bear_score > bull_score:
        overall = "медвежий"
    
    return {
        "overall": overall,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "total_score": bull_score - bear_score,
        "patterns": active_patterns
    }


if __name__ == "__main__":
    try:
        # Проверяем наличие файла данных
        test_file = os.path.join(DATA_DIR, "test_candlestick_data.json")
        
        # Если нет файла, используем любой другой в директории
        if not os.path.exists(test_file):
            files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            if files:
                test_file = os.path.join(DATA_DIR, files[0])
                print(f"Используем файл: {test_file}")
            else:
                raise APK.DatabaseError("Не найдены файлы JSON для анализа")
        
        # Загружаем данные
        data = pd.read_json(test_file)
        
        # Анализируем паттерны
        patterns = current_candlestick_patterns(data)
        
        # Выводим результаты последних 10 свечей
        recent_patterns = patterns.tail(10)
        print("\nОбнаруженные паттерны (последние 10 свечей):")
        
        for idx, row in recent_patterns.iterrows():
            pattern_info = []
            for col in row.index:
                if row[col] != 0 and row[col] is not False:
                    if isinstance(row[col], str):
                        pattern_info.append(f"{col}: {row[col]}")
                    else:
                        pattern_info.append(f"{col}: {row[col]:.2f}")
            
            if pattern_info:
                print(f"Индекс {idx}: {', '.join(pattern_info)}")
        
        # Получаем сводную оценку
        signals = get_pattern_signals(recent_patterns)
        
        print("\nИтоговая оценка:")
        print(f"Рекомендация: {signals['overall']}")
        print(f"Бычий счет: {signals['bull_score']:.2f}")
        print(f"Медвежий счет: {signals['bear_score']:.2f}")
        print(f"Общий счет: {signals['total_score']:.2f}")
        
        if signals['patterns']:
            print("\nАктивные паттерны:")
            for p in signals['patterns']:
                print(f"  - {p['name']} ({p['type']}, уверенность: {p['confidence']:.2f})")
        
    except Exception as e:
        print(f"Ошибка при анализе паттернов: {e}")