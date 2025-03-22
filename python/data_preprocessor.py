import os
import pandas as pd
import numpy as np
import datetime
from typing import Dict, List, Optional, Union, Tuple
import _AppProjectKit as APK

# Путь к директории для хранения данных
DATA_DIR = APK.DATA_DIR


def load_data(filename: str) -> pd.DataFrame:
    """
    Загружает данные из указанного JSON-файла в DataFrame.
    
    Аргументы:
        filename: Имя файла или полный путь к нему
        
    Возвращает:
        DataFrame с загруженными данными
        
    Вызывает:
        FileNotFoundError: если файл не найден
        APK.InvalidInputError: если формат данных не соответствует ожидаемому
    """
    filepath = filename if os.path.dirname(filename) else os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filename} не найден в директории {DATA_DIR}.")
    
    try:
        data = pd.read_json(filepath)
        print(f"Данные загружены из файла: {filename}")
        return data
    except Exception as e:
        raise APK.InvalidInputError(f"Ошибка при чтении файла {filename}: {str(e)}")


def preprocess_data(data: pd.DataFrame, options: Optional[Dict] = None) -> pd.DataFrame:
    """
    Применяет предобработку к данным.
    
    Аргументы:
        data: Исходный DataFrame
        options: Словарь с параметрами предобработки:
            - 'handle_missing': способ обработки пропущенных значений ('drop', 'interpolate', 'ffill', 'mean')
            - 'normalize': нормализовать ли числовые столбцы (True/False)
            - 'remove_outliers': удалять ли выбросы (True/False)
            - 'outlier_std': количество стандартных отклонений для определения выбросов
            - 'volume_filter': фильтровать ли по объему (минимальное значение)
        
    Возвращает:
        DataFrame с предобработанными данными
    """
    # Параметры по умолчанию
    default_options = {
        'handle_missing': 'interpolate',
        'normalize': False,
        'remove_outliers': False,
        'outlier_std': 3.0,
        'volume_filter': None
    }
    
    # Объединяем с пользовательскими параметрами
    if options:
        default_options.update(options)
    
    # Создаем копию данных
    data_cleaned = data.copy()
    
    # Проверяем наличие обязательных столбцов
    required_columns = ['close']
    missing_columns = [col for col in required_columns if col not in data_cleaned.columns]
    if missing_columns:
        raise APK.InvalidInputError(f"Отсутствуют обязательные столбцы: {', '.join(missing_columns)}")
    
    # Обработка пропущенных значений
    if default_options['handle_missing'] == 'drop':
        data_cleaned = data_cleaned.dropna()
    elif default_options['handle_missing'] == 'interpolate':
        data_cleaned = data_cleaned.interpolate(method='linear')
    elif default_options['handle_missing'] == 'ffill':
        data_cleaned = data_cleaned.fillna(method='ffill')
    elif default_options['handle_missing'] == 'mean':
        # Для числовых столбцов используем среднее, для категориальных - наиболее частое значение
        for col in data_cleaned.columns:
            if pd.api.types.is_numeric_dtype(data_cleaned[col]):
                data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mean())
            else:
                data_cleaned[col] = data_cleaned[col].fillna(data_cleaned[col].mode()[0])
    
    # Удаление выбросов
    if default_options['remove_outliers']:
        numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['close', 'high', 'low', 'open', 'volume']:  # только для ценовых столбцов
                mean = data_cleaned[col].mean()
                std = data_cleaned[col].std()
                threshold = default_options['outlier_std'] * std
                data_cleaned = data_cleaned[
                    (data_cleaned[col] >= mean - threshold) & 
                    (data_cleaned[col] <= mean + threshold)
                ]
    
    # Фильтрация по объему
    if default_options['volume_filter'] and 'volume' in data_cleaned.columns:
        data_cleaned = data_cleaned[data_cleaned['volume'] >= default_options['volume_filter']]
    
    # Нормализация (масштабирование к диапазону 0-1)
    if default_options['normalize']:
        numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['close', 'high', 'low', 'open']:  # только для ценовых столбцов
                min_val = data_cleaned[col].min()
                max_val = data_cleaned[col].max()
                if max_val > min_val:  # избегаем деления на ноль
                    data_cleaned[f'{col}_normalized'] = (data_cleaned[col] - min_val) / (max_val - min_val)
    
    print(f"Предобработка данных завершена. Исходно строк: {len(data)}, после обработки: {len(data_cleaned)}")
    return data_cleaned


def add_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет технические индикаторы и признаки к данным.
    
    Аргументы:
        data: Исходный DataFrame
        
    Возвращает:
        DataFrame с добавленными признаками
    """
    df = data.copy()
    
    # Проверка наличия необходимых столбцов
    required_columns = ['close', 'high', 'low']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise APK.InvalidInputError(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
    
    # Добавляем сдвиги цен закрытия
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    
    # Добавляем процентные изменения цен
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    
    # Добавляем среднее значение для разных окон
    for window in [5, 10, 20]:
        df[f'close_ma_{window}'] = df['close'].rolling(window=window).mean()
    
    # Добавляем стандартное отклонение (волатильность)
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
    
    # Диапазон дня
    df['day_range'] = df['high'] - df['low']
    df['day_range_pct'] = df['day_range'] / df['close']
    
    # Расстояние от минимума и максимума
    df['close_to_high'] = (df['high'] - df['close']) / df['close']
    df['close_to_low'] = (df['close'] - df['low']) / df['close']
    
    # Заполняем NaN нулями или удаляем первые строки с NaN
    df = df.iloc[20:]  # Удаляем первые 20 строк, так как некоторые признаки используют окно 20
    
    print(f"Добавлены технические признаки. Новое количество столбцов: {len(df.columns)}")
    return df


def save_preprocessed_data(data: pd.DataFrame, original_filename: str) -> str:
    """
    Сохраняет предобработанные данные в новый JSON-файл.
    
    Аргументы:
        data: DataFrame с предобработанными данными
        original_filename: Оригинальное имя файла
        
    Возвращает:
        Путь к сохраненному файлу
    """
    # Генерируем новый путь для сохранения
    timestamp = datetime.datetime.now().strftime("[%H%M%S]")
    new_filename = f"preprocessed_{os.path.basename(original_filename).replace('.json', '')}_{timestamp}.json"
    new_filepath = os.path.join(DATA_DIR, new_filename)
    
    # Добавляем метаданные
    metadata = {
        "original_file": original_filename,
        "preprocessing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rows_count": len(data),
        "columns": list(data.columns)
    }
    
    # Подготавливаем данные для сохранения
    output = {
        "metadata": metadata,
        "data": data.to_dict(orient='records')
    }
    
    # Сохраняем в JSON
    with open(new_filepath, 'w', encoding='utf-8') as f:
        import json
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"Предобработанные данные сохранены в файл: {new_filepath}")
    return new_filepath


def aggregate_data(data: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """
    Агрегирует данные по заданной частоте (например, из минутных в дневные).
    
    Аргументы:
        data: Исходный DataFrame с временным рядом
        freq: Частота агрегации ('D' - день, 'W' - неделя, 'M' - месяц и т.д.)
        
    Возвращает:
        DataFrame с агрегированными данными
    """
    # Проверяем, есть ли столбец с датой
    date_column = None
    for col in data.columns:
        if col in ['date', 'begin', 'time', 'datetime']:
            date_column = col
            break
    
    if not date_column:
        raise APK.InvalidInputError("Не найден столбец с датой для агрегации")
    
    # Проверяем формат даты и конвертируем при необходимости
    if not pd.api.types.is_datetime64_dtype(data[date_column]):
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    
    # Устанавливаем дату как индекс
    data_indexed = data.set_index(date_column)
    
    # Агрегируем данные
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Используем только столбцы, которые есть в данных
    available_columns = {col: agg_dict[col] for col in agg_dict if col in data_indexed.columns}
    
    # Выполняем агрегацию
    aggregated = data_indexed.resample(freq).agg(available_columns)
    
    # Сбрасываем индекс
    aggregated = aggregated.reset_index()
    
    print(f"Данные агрегированы по частоте {freq}. Новое количество строк: {len(aggregated)}")
    return aggregated


def process_file(filename: str, 
                process_options: Optional[Dict] = None,
                add_features: bool = False,
                aggregate: Optional[str] = None) -> str:
    """
    Полный процесс обработки файла: загрузка, предобработка, добавление признаков, сохранение.
    
    Аргументы:
        filename: Имя файла для обработки
        process_options: Параметры предобработки
        add_features: Добавлять ли технические признаки
        aggregate: Частота агрегации (None, если агрегация не нужна)
        
    Возвращает:
        Путь к сохраненному файлу с обработанными данными
    """
    try:
        # Загрузка данных
        data = load_data(filename)
        
        # Базовая предобработка
        processed_data = preprocess_data(data, process_options)
        
        # Агрегация (если требуется)
        if aggregate:
            processed_data = aggregate_data(processed_data, freq=aggregate)
        
        # Добавление технических признаков (если требуется)
        if add_features:
            processed_data = add_technical_features(processed_data)
        
        # Сохранение результатов
        return save_preprocessed_data(processed_data, filename)
        
    except Exception as e:
        print(f"Ошибка при обработке файла {filename}: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Демонстрация использования функций модуля
        import sys
        
        if len(sys.argv) > 1:
            # Если указан файл в аргументах командной строки
            filename = sys.argv[1]
            process_file(filename, add_features=True)
        else:
            # Тестовый пример
            test_data = pd.DataFrame({
                'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
                'open': np.random.normal(100, 5, 100),
                'high': np.random.normal(105, 5, 100),
                'low': np.random.normal(95, 5, 100),
                'close': np.random.normal(102, 5, 100),
                'volume': np.random.randint(1000, 10000, 100)
            })
            
            # Добавляем немного пропущенных значений
            test_data.loc[10:15, 'close'] = np.nan
            
            # Сохраняем тестовые данные
            test_filename = "test_data_for_preprocessing.json"
            test_filepath = os.path.join(DATA_DIR, test_filename)
            test_data.to_json(test_filepath)
            
            # Обрабатываем тестовый файл
            process_file(test_filename, 
                       process_options={'handle_missing': 'interpolate', 'remove_outliers': True},
                       add_features=True)
            
            # Удаляем тестовый файл
            os.remove(test_filepath)
            
    except APK.ApplicationError as e:
        print(f"Ошибка приложения: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")