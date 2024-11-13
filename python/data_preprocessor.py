import os
import pandas as pd
import json
import datetime

# Путь к директории для хранения данных
DATA_DIR = "storage"

def load_data(filename: str) -> pd.DataFrame:
    """
    Загружает данные из указанного JSON-файла в DataFrame.
    """
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filename} не найден в директории {DATA_DIR}.")
    
    data = pd.read_json(filepath)
    print(f"Данные загружены из файла: {filename}")
    return data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет предобработку к данным.
    Например, фильтрация NaN значений и нормализация столбцов.
    """
    # Пример предобработки: удаление записей с пустыми значениями
    data_cleaned = data.dropna()

    # Пример нормализации (если нужно): нормализуем цены относительно максимального значения
    if 'close' in data_cleaned.columns:
        max_close = data_cleaned['close'].max()
        if max_close != 0:
            data_cleaned['close_normalized'] = data_cleaned['close'] / max_close

    print("Предобработка данных завершена.")
    return data_cleaned

def save_preprocessed_data(data: pd.DataFrame, original_filename: str) -> None:
    """
    Сохраняет предобработанные данные в новый JSON-файл.
    """
    # Генерируем новый путь для сохранения
    timestamp = datetime.datetime.now().strftime("[%H%M%S]")
    new_filename = f"preprocessed_{original_filename.replace('.json', '')}_{timestamp}.json"
    new_filepath = os.path.join(DATA_DIR, new_filename)

    # Сохраняем предобработанные данные в новый JSON-файл
    data.to_json(new_filepath, indent=4, force_ascii=False)
    print(f"Предобработанные данные сохранены в файл: {new_filepath}")

def process_file(filename: str) -> None:
    """
    Загружает, предобрабатывает и сохраняет данные из указанного файла.
    """
    data = load_data(filename)

    processed_data = preprocess_data(data)

    save_preprocessed_data(processed_data, filename)


