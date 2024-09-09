import os
import json
import pandas as pd
from typing import List, Dict, Any

# Путь к директории для хранения данных
DATA_DIR = "storage"

# Проверяем наличие директории, если она не существует — создаем
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def list_files(extension: str = ".json") -> List[str]:
    """
    Перечисляет все файлы в директории DATA_DIR с указанным расширением.
    По умолчанию — JSON файлы.
    """
    return [f for f in os.listdir(DATA_DIR) if f.endswith(extension)]

def load_json(filename: str) -> pd.DataFrame:
    """
    Загружает данные из указанного JSON-файла и возвращает как DataFrame.
    """
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filename} не найден в директории {DATA_DIR}.")
    
    data = pd.read_json(filepath)
    print(f"Данные загружены из файла: {filename}")
    return data

def save_json(data: pd.DataFrame, filename: str) -> None:
    """
    Сохраняет переданный DataFrame в JSON-файл с указанным именем.
    """
    filepath = os.path.join(DATA_DIR, filename)
    data.to_json(filepath, indent=4, force_ascii=False)
    print(f"Данные сохранены в файл: {filename}")

def delete_file(filename: str) -> None:
    """
    Удаляет указанный файл из директории DATA_DIR.
    """
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"Файл {filename} удален.")
    else:
        print(f"Файл {filename} не найден.")

def load_all_json() -> List[Dict[str, Any]]:
    """
    Загружает все JSON-файлы из директории DATA_DIR и возвращает их содержимое.
    """
    files = list_files(".json")
    all_data = []
    for file in files:
        filepath = os.path.join(DATA_DIR, file)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.append({file: data})
    return all_data


