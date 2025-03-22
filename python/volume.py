import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Константы для анализа объемов
VOLUME_MA_PERIOD = 20      # Период для скользящей средней объема
VOLUME_THRESHOLD = 1.5     # Множитель для определения высокого объема
STRONG_SIGNAL_THRESHOLD = 2.0  # Множитель для определения очень высокого объема


def load_data(filename: str) -> pd.DataFrame:
    """
    Загружает данные из JSON-файла в DataFrame.
    
    Аргументы:
        filename: Имя файла или полный путь к нему
        
    Возвращает:
        DataFrame с загруженными данными
        
    Вызывает:
        APK.DatabaseError: при ошибке доступа к файлу
        APK.InvalidInputError: при ошибке в структуре данных
    """
    try:
        # Формируем полный путь
        full_path = filename if os.path.dirname(filename) else os.path.join(DATA_DIR, filename)
        
        # Проверяем существование файла
        if not os.path.exists(full_path):
            raise APK.DatabaseError(f"Файл данных не найден: {filename}")
        
        # Пытаемся загрузить данные
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Определяем структуру данных
        if isinstance(data, dict) and all(key in data for key in ['begin', 'close', 'volume']):
            # Структура данных FEES
            df = pd.DataFrame({
                'date': list(data['begin'].values()),
                'close': list(data['close'].values()),
                'volume': list(data['volume'].values())
            })
        else:
            # Обычная структура данных
            df = pd.DataFrame(data)
        
        # Проверка наличия необходимых столбцов
        required_columns = ['close', 'volume']
        if 'date' not in df.columns and 'begin' in df.columns:
            df['date'] = df['begin']  # Используем 'begin' как 'date'
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        if 'date' not in df.columns and 'begin' not in df.columns:
            missing_columns.append('date или begin')
            
        if missing_columns:
            raise APK.InvalidInputError(f"Отсутствуют обязательные столбцы: {', '.join(missing_columns)}")
        
        # Преобразование типов данных
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Преобразование даты, если она есть
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Проверка на пропущенные значения
        if df['close'].isna().any() or df['volume'].isna().any():
            print(f"Предупреждение: в данных обнаружены пропущенные значения. Заполняем...")
            df = df.interpolate(method='linear')
        
        return df
        
    except FileNotFoundError:
        raise APK.DatabaseError(f"Файл данных не найден: {filename}")
    except json.JSONDecodeError:
        raise APK.InvalidInputError(f"Ошибка при чтении JSON-файла: {filename}")
    except ValueError as e:
        raise APK.InvalidInputError(f"Ошибка преобразования данных: {str(e)}")
    except Exception as e:
        raise APK.ApplicationError(f"Непредвиденная ошибка при загрузке данных: {str(e)}")


def find_support_resistance(df: pd.DataFrame, lookback: int = 20) -> APK.todaySupRes:
    """
    Находит уровни поддержки и сопротивления используя кластеризацию цен.
    
    Аргументы:
        df: DataFrame с ценовыми данными
        lookback: Количество последних баров для анализа
        
    Возвращает:
        Объект todaySupRes с уровнями поддержки и сопротивления
    """
    try:
        # Получаем последние данные
        recent_data = df.tail(lookback)
        
        # Кластеризуем цены для более точного определения уровней
        prices = recent_data['close'].tolist()
        prices.sort()
        
        # Находим кластеры цен (близко расположенные уровни)
        clusters = []
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            # Если цена близка к средней в текущем кластере (в пределах 0.5%)
            if abs(price - sum(current_cluster) / len(current_cluster)) / price < 0.005:
                current_cluster.append(price)
            else:
                # Добавляем среднее значение кластера
                if current_cluster:
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [price]
        
        # Добавляем последний кластер
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        # Определяем уровни на основе кластеров
        if len(clusters) >= 4:
            # Берем крайние и промежуточные уровни
            clusters.sort()
            
            support_3 = clusters[0]
            support_2 = clusters[len(clusters) // 4]
            support_1 = clusters[len(clusters) // 2 - 1]
            pivot = clusters[len(clusters) // 2]
            resistance_1 = clusters[len(clusters) // 2 + 1]
            resistance_2 = clusters[3 * len(clusters) // 4]
            resistance_3 = clusters[-1]
        else:
            # Если кластеров мало, используем минимум, максимум и среднее
            support_3 = recent_data['close'].min() * 0.9
            support_2 = recent_data['close'].min() * 0.95
            support_1 = recent_data['close'].min()
            pivot = recent_data['close'].mean()
            resistance_1 = recent_data['close'].max()
            resistance_2 = recent_data['close'].max() * 1.05
            resistance_3 = recent_data['close'].max() * 1.1
        
        return APK.todaySupRes(
            pivot=pivot,
            resistance_1=resistance_1,
            resistance_2=resistance_2,
            resistance_3=resistance_3,
            support_1=support_1,
            support_2=support_2,
            support_3=support_3
        )
    except Exception as e:
        raise APK.InvalidInputError(f"Ошибка при расчете уровней: {str(e)}")


def detect_volume_anomalies(df: pd.DataFrame, window: int = VOLUME_MA_PERIOD) -> pd.DataFrame:
    """
    Обнаруживает аномалии в объемах торгов.
    
    Аргументы:
        df: DataFrame с данными
        window: Размер окна для скользящего среднего
        
    Возвращает:
        DataFrame с добавленными индикаторами аномалий
    """
    result = df.copy()
    
    # Рассчитываем скользящее среднее объема
    result['volume_ma'] = result['volume'].rolling(window=window).mean()
    
    # Рассчитываем стандартное отклонение объема
    result['volume_std'] = result['volume'].rolling(window=window).std()
    
    # Z-оценка для объема (сколько стандартных отклонений от среднего)
    result['volume_z'] = (result['volume'] - result['volume_ma']) / result['volume_std']
    
    # Определяем аномалии (значения выше 2 стандартных отклонений)
    result['volume_anomaly'] = result['volume_z'] > 2
    
    # Классифицируем аномалии по силе
    result['anomaly_strength'] = 0
    result.loc[result['volume_z'] > 2, 'anomaly_strength'] = 1
    result.loc[result['volume_z'] > 3, 'anomaly_strength'] = 2
    result.loc[result['volume_z'] > 4, 'anomaly_strength'] = 3
    
    return result


def calculate_volume_price_correlation(df: pd.DataFrame, window: int = VOLUME_MA_PERIOD) -> pd.DataFrame:
    """
    Рассчитывает корреляцию между объемом и изменением цены.
    
    Аргументы:
        df: DataFrame с данными
        window: Размер окна для расчета корреляции
        
    Возвращает:
        DataFrame с добавленной корреляцией
    """
    result = df.copy()
    
    # Рассчитываем процентное изменение цены
    result['price_change'] = result['close'].pct_change()
    
    # Нормализуем объем
    result['volume_normalized'] = result['volume'] / result['volume'].rolling(window=window).mean()
    
    # Рассчитываем корреляцию между объемом и абсолютным изменением цены
    result['price_change_abs'] = result['price_change'].abs()
    
    # Используем скользящую корреляцию Спирмена (более устойчива к выбросам)
    result['volume_price_corr'] = (
        result['volume_normalized'].rolling(window=window)
        .corr(result['price_change_abs'])
    )
    
    # Определяем, подтверждает ли объем движение цены
    # Высокая корреляция означает, что большие изменения цены происходят на большом объеме
    result['volume_confirms_price'] = result['volume_price_corr'] > 0.7
    
    return result


def volume_analysis(df: pd.DataFrame, support: float, resistance: float) -> pd.DataFrame:
    """
    Анализирует объемы и цены, возвращает DataFrame с сигналами.
    
    Сигналы:
    2  = Сильный сигнал на покупку (прорыв сопротивления с высоким объемом)
    1  = Слабый сигнал на покупку (рост цены при высоком объеме)
    0  = Нет сигнала
    -1 = Слабый сигнал на продажу (падение цены при высоком объеме)
    -2 = Сильный сигнал на продажу (прорыв поддержки с высоким объемом)
    
    Аргументы:
        df: DataFrame с данными цен и объемов
        support: Уровень поддержки
        resistance: Уровень сопротивления
        
    Возвращает:
        DataFrame с добавленными сигналами
    """
    # Добавляем обнаружение аномалий объема
    df_with_anomalies = detect_volume_anomalies(df)
    
    # Добавляем корреляцию объема и цены
    result = calculate_volume_price_correlation(df_with_anomalies)
    
    # Добавляем предыдущие значения для сравнения
    result['prev_close'] = result['close'].shift(1)
    result['prev_volume'] = result['volume'].shift(1)
    
    # Определяем высокий объем (выше скользящего среднего)
    result['high_volume'] = result['volume'] > result['volume_ma'] * VOLUME_THRESHOLD
    
    # Определяем очень высокий объем
    result['very_high_volume'] = result['volume'] > result['volume_ma'] * STRONG_SIGNAL_THRESHOLD
    
    # Инициализируем сигналы
    result['Volume_Signal'] = 0
    
    # Сильный сигнал на покупку (прорыв сопротивления с очень высоким объемом)
    result.loc[
        (result['close'] > resistance) & 
        (result['very_high_volume']), 
        'Volume_Signal'
    ] = 2
    
    # Сильный сигнал на продажу (прорыв поддержки с очень высоким объемом)
    result.loc[
        (result['close'] < support) & 
        (result['very_high_volume']), 
        'Volume_Signal'
    ] = -2
    
    # Слабый сигнал на покупку (рост цены при высоком объеме)
    result.loc[
        (result['close'] > result['prev_close']) & 
        (result['high_volume']) & 
        (result['Volume_Signal'] == 0), 
        'Volume_Signal'
    ] = 1
    
    # Слабый сигнал на продажу (падение цены при высоком объеме)
    result.loc[
        (result['close'] < result['prev_close']) & 
        (result['high_volume']) & 
        (result['Volume_Signal'] == 0), 
        'Volume_Signal'
    ] = -1
    
    # Добавляем фильтр для уменьшения ложных сигналов
    # Если корреляция объема и цены низкая, снижаем силу сигнала
    result.loc[
        (result['volume_confirms_price'] == False) & 
        (result['Volume_Signal'] != 0), 
        'Volume_Signal'
    ] = result['Volume_Signal'] / 2
    
    return result


def get_volume_summary(data: pd.DataFrame) -> Dict[str, Union[str, float, int]]:
    """
    Создает сводку по текущим значениям объемов и сигналам
    
    Аргументы:
        data: DataFrame с рассчитанными показателями объема
        
    Возвращает:
        Словарь с текущим состоянием индикатора
    """
    try:
        if len(data) < 2:
            raise APK.InvalidInputError("Недостаточно данных для анализа объемов")
            
        # Получаем последние два значения
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Определяем тренд объема
        volume_change_pct = (latest['volume'] - prev['volume']) / prev['volume'] * 100
        
        if volume_change_pct > 20:
            volume_trend = "Резко растущий"
        elif volume_change_pct > 5:
            volume_trend = "Растущий"
        elif volume_change_pct < -20:
            volume_trend = "Резко падающий"
        elif volume_change_pct < -5:
            volume_trend = "Падающий"
        else:
            volume_trend = "Боковой"
            
        # Расшифровка сигнала
        signal_desc = {
            2: "Сильный сигнал на покупку (прорыв сопротивления с высоким объемом)",
            1: "Слабый сигнал на покупку (рост цены при высоком объеме)",
            0: "Нет сигнала",
            -1: "Слабый сигнал на продажу (падение цены при высоком объеме)",
            -2: "Сильный сигнал на продажу (прорыв поддержки с высоким объемом)"
        }
        
        # Округляем сигнал до ближайшего целого для определения описания
        signal_value = latest['Volume_Signal']
        signal_key = round(signal_value)
        
        # Оцениваем аномалии объема, если доступны
        anomaly_desc = ""
        if 'anomaly_strength' in latest:
            if latest['anomaly_strength'] == 3:
                anomaly_desc = "Крайне высокий объем (более 4 стандартных отклонений)"
            elif latest['anomaly_strength'] == 2:
                anomaly_desc = "Очень высокий объем (более 3 стандартных отклонений)"
            elif latest['anomaly_strength'] == 1:
                anomaly_desc = "Высокий объем (более 2 стандартных отклонений)"
                
        # Добавляем информацию о корреляции объема и цены
        correlation_desc = ""
        if 'volume_price_corr' in latest and not pd.isna(latest['volume_price_corr']):
            corr = latest['volume_price_corr']
            if corr > 0.8:
                correlation_desc = "Объем сильно подтверждает движение цены"
            elif corr > 0.5:
                correlation_desc = "Объем умеренно подтверждает движение цены"
            elif corr < 0.2:
                correlation_desc = "Объем слабо связан с движением цены"
        
        # Формируем итоговую сводку
        summary = {
            'volume_trend': volume_trend,
            'current_volume': latest['volume'],
            'prev_volume': prev['volume'],
            'volume_change': volume_change_pct,
            'signal': signal_value,
            'signal_description': signal_desc.get(signal_key, "Неопределенный сигнал"),
            'anomaly_description': anomaly_desc,
            'correlation_description': correlation_desc
        }
        
        # Добавляем данные о скользящем среднем объема, если доступно
        if 'volume_ma' in latest and not pd.isna(latest['volume_ma']):
            summary['volume_ma'] = latest['volume_ma']
            summary['volume_to_ma_ratio'] = latest['volume'] / latest['volume_ma']
        
        return summary
        
    except Exception as e:
        raise APK.ApplicationError(f"Ошибка при создании сводки объемов: {str(e)}")


def identify_volume_patterns(df: pd.DataFrame, window: int = VOLUME_MA_PERIOD) -> Dict[str, List[int]]:
    """
    Идентифицирует паттерны объема и их позиции в DataFrame.
    
    Аргументы:
        df: DataFrame с данными
        window: Размер окна для анализа
        
    Возвращает:
        Словарь с найденными паттернами и их индексами
    """
    result = detect_volume_anomalies(df)
    
    # Инициализируем словарь паттернов
    patterns = {
        'volume_climax': [],       # Объемный максимум (потенциальный разворот)
        'volume_breakout': [],     # Прорыв объема (потенциальное начало тренда)
        'volume_dryup': [],        # Иссякающий объем (потенциальный конец тренда)
        'rising_volume': [],       # Растущий объем (усиление тренда)
        'falling_volume': []       # Падающий объем (ослабление тренда)
    }
    
    for i in range(window, len(result)):
        # Оконные данные для анализа
        window_data = result.iloc[i-window:i+1]
        
        # Последние значения
        last_volume = window_data['volume'].iloc[-1]
        volume_ma = window_data['volume_ma'].iloc[-1]
        
        # Проверка на объемный максимум
        if last_volume == window_data['volume'].max() and last_volume > volume_ma * 2:
            patterns['volume_climax'].append(i)
        
        # Проверка на прорыв объема после периода низкого объема
        if (last_volume > volume_ma * 1.5 and 
            all(window_data['volume'].iloc[-4:-1] < volume_ma)):
            patterns['volume_breakout'].append(i)
        
        # Проверка на иссякающий объем
        if (last_volume < volume_ma * 0.7 and 
            all(window_data['volume'].iloc[-4:-1] > window_data['volume'].iloc[-1])):
            patterns['volume_dryup'].append(i)
        
        # Проверка на последовательно растущий объем
        if all(window_data['volume'].iloc[j] < window_data['volume'].iloc[j+1] for j in range(-4, -1)):
            patterns['rising_volume'].append(i)
        
        # Проверка на последовательно падающий объем
        if all(window_data['volume'].iloc[j] > window_data['volume'].iloc[j+1] for j in range(-4, -1)):
            patterns['falling_volume'].append(i)
    
    return patterns


def main(filename: str) -> None:
    """
    Основная функция для анализа объемов.
    
    Аргументы:
        filename: Имя файла с данными
    """
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
        
        # Идентификация паттернов объема
        patterns = identify_volume_patterns(result)
        
        # Вывод результатов
        print("\nПоследние 10 значений объемов и сигналов:")
        print(result[['close', 'volume', 'volume_ma', 'Volume_Signal']].tail(10))
        
        print("\nТекущее состояние объемов:")
        print(f"Тренд объема: {summary['volume_trend']}")
        print(f"Текущий объем: {summary['current_volume']:,.0f}")
        print(f"Изменение объема: {summary['volume_change']:.2f}%")
        print(f"Сигнал ({summary['signal']}): {summary['signal_description']}")
        
        if summary.get('anomaly_description'):
            print(f"Аномалия: {summary['anomaly_description']}")
            
        if summary.get('correlation_description'):
            print(f"Корреляция: {summary['correlation_description']}")
        
        # Вывод обнаруженных паттернов
        print("\nОбнаруженные паттерны объема (последние 20 баров):")
        recent_patterns = []
        bar_count = min(20, len(df))
        recent_range = list(range(len(df) - bar_count, len(df)))
        
        for pattern_name, indices in patterns.items():
            recent_indices = [i for i in indices if i in recent_range]
            if recent_indices:
                positions = [len(df) - i for i in reversed(recent_indices)]
                recent_patterns.append(f"{pattern_name}: {positions} баров назад")
        
        if recent_patterns:
            for pattern in recent_patterns:
                print(f"- {pattern}")
        else:
            print("Явных паттернов объема не обнаружено")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")


if __name__ == "__main__":
    try:
        # Проверяем наличие файла
        default_filename = "FEES_2024-11-10_3D_[183522].json"
        
        if os.path.exists(os.path.join(DATA_DIR, default_filename)):
            main(default_filename)
        else:
            # Ищем любой JSON-файл в директории
            json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            if json_files:
                main(json_files[0])
            else:
                print(f"В директории {DATA_DIR} не найдено JSON-файлов с данными.")
            
    except Exception as e:
        print(f"Ошибка при запуске: {e}")