import os
import json
import pandas as pd
import datetime
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
import _AppProjectKit as APK

# Константы и настройки
# =====================
DATA_DIR = APK.DATA_DIR

# Проверяем наличие директории, если она не существует — создаем
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Словарь для индексирования файлов по тикерам для быстрого доступа
# {ticker: [filepath1, filepath2, ...]}
file_index: Dict[str, List[str]] = {}


def build_file_index() -> None:
    """
    Строит индекс файлов по тикерам для быстрого поиска.
    Вызывается при инициализации модуля и после операций с файлами.
    """
    global file_index
    file_index.clear()
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            try:
                # Предполагаем, что имя файла начинается с тикера
                ticker = filename.split('_')[0]
                if ticker not in file_index:
                    file_index[ticker] = []
                file_index[ticker].append(os.path.join(DATA_DIR, filename))
            except Exception:
                # Пропускаем файл, если не удалось определить тикер
                pass


def list_files(extension: str = ".json", ticker: Optional[str] = None) -> List[str]:
    """
    Перечисляет все файлы в директории DATA_DIR с указанным расширением.
    Может фильтровать по тикеру.
    
    Аргументы:
        extension: Расширение файлов (по умолчанию — JSON)
        ticker: Тикер для фильтрации (опционально)
        
    Возвращает:
        Список подходящих файлов
    """
    if ticker and ticker in file_index:
        return [f for f in file_index[ticker] if f.endswith(extension)]
    
    return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
            if f.endswith(extension) and (ticker is None or f.startswith(ticker))]


def load_json(filename: str) -> pd.DataFrame:
    """
    Загружает данные из указанного JSON-файла и возвращает как DataFrame.
    Поддерживает чтение как прямых JSON файлов, так и файлов с метаданными.
    
    Аргументы:
        filename: Имя файла или полный путь к нему
        
    Возвращает:
        DataFrame с данными
        
    Вызывает:
        FileNotFoundError: если файл не найден
        APK.InvalidInputError: если формат данных не соответствует ожидаемому
    """
    filepath = filename if os.path.dirname(filename) else os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл {filename} не найден в директории {DATA_DIR}.")
    
    try:
        # Пробуем загрузить данные напрямую как DataFrame
        data = pd.read_json(filepath)
        
        # Проверяем, пришли ли данные как DataFrame или словарь
        if isinstance(data, dict) and "metadata" in data and "data" in data:
            # Если это файл с метаданными, извлекаем данные
            data = pd.DataFrame(data["data"])
        
        # Проверяем наличие необходимых столбцов
        if 'close' not in data.columns:
            raise APK.InvalidInputError(f"В файле {filename} отсутствуют необходимые столбцы (close)")
            
        # Проверяем типы данных
        if not pd.api.types.is_numeric_dtype(data['close']):
            data['close'] = pd.to_numeric(data['close'], errors='coerce')
            
        # Преобразуем дату/время, если это необходимо
        if 'date' in data.columns and not pd.api.types.is_datetime64_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            
        print(f"Данные загружены из файла: {filename}")
        return data
        
    except pd.errors.ParserError:
        # Если не удалось загрузить напрямую, пробуем через json.load
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            # Проверяем, есть ли метаданные
            if isinstance(raw_data, dict) and "metadata" in raw_data and "data" in raw_data:
                data = pd.DataFrame(raw_data["data"])
            else:
                data = pd.DataFrame(raw_data)
                
            print(f"Данные загружены из файла (с преобразованием): {filename}")
            return data
            
        except json.JSONDecodeError:
            raise APK.InvalidInputError(f"Ошибка декодирования JSON в файле {filename}")
    
    except Exception as e:
        raise APK.DatabaseError(f"Ошибка при загрузке файла {filename}: {str(e)}")


def save_json(data: pd.DataFrame, 
              filename: str, 
              add_metadata: bool = True, 
              ticker: Optional[str] = None) -> str:
    """
    Сохраняет переданный DataFrame в JSON-файл с указанным именем.
    
    Аргументы:
        data: DataFrame для сохранения
        filename: Имя файла или полный путь к нему
        add_metadata: Добавлять ли метаданные (дата сохранения, размер и т. д.)
        ticker: Тикер для метаданных (по умолчанию берется из имени файла)
        
    Возвращает:
        Путь к сохраненному файлу
    """
    filepath = filename if os.path.dirname(filename) else os.path.join(DATA_DIR, filename)
    
    try:
        # Проверка целостности данных перед сохранением
        if data.empty:
            raise APK.InvalidInputError("Невозможно сохранить пустой DataFrame")
        
        # Добавляем метаданные, если требуется
        if add_metadata:
            # Определяем тикер из имени файла, если не указан
            if ticker is None:
                ticker = os.path.basename(filepath).split('_')[0]
                
            # Создаем метаданные
            metadata = {
                "ticker": ticker,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows_count": len(data),
                "columns": list(data.columns),
                "checksum": hashlib.md5(data.to_json().encode()).hexdigest()
            }
            
            # Формируем структуру с метаданными
            output = {
                "metadata": metadata,
                "data": data.to_dict(orient='records')
            }
            
            # Сохраняем в JSON с метаданными
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        else:
            # Прямое сохранение DataFrame
            data.to_json(filepath, indent=4, force_ascii=False, orient='records')
            
        print(f"Данные сохранены в файл: {filepath}")
        
        # Обновляем индекс файлов
        build_file_index()
        
        return filepath
        
    except Exception as e:
        raise APK.DatabaseError(f"Ошибка при сохранении в файл {filepath}: {str(e)}")


def delete_file(filename: str) -> None:
    """
    Удаляет указанный файл из директории DATA_DIR.
    
    Аргументы:
        filename: Имя файла или полный путь к нему
    """
    filepath = filename if os.path.dirname(filename) else os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Файл {filepath} удален.")
            
            # Обновляем индекс файлов
            build_file_index()
            
        except Exception as e:
            raise APK.DatabaseError(f"Ошибка при удалении файла {filepath}: {str(e)}")
    else:
        print(f"Файл {filepath} не найден.")


def find_latest_data(ticker: str, days_limit: int = 30) -> Optional[pd.DataFrame]:
    """
    Находит самые свежие данные для указанного тикера.
    
    Аргументы:
        ticker: Тикер инструмента
        days_limit: Ограничение по дням (искать файлы не старше)
        
    Возвращает:
        DataFrame с данными или None, если данные не найдены
    """
    if ticker not in file_index:
        build_file_index()
        if ticker not in file_index:
            return None
            
    # Фильтруем файлы по дате создания
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_limit)
    files = []
    
    for filepath in file_index[ticker]:
        try:
            file_stat = os.stat(filepath)
            file_date = datetime.datetime.fromtimestamp(file_stat.st_mtime)
            if file_date >= cutoff_date:
                files.append((filepath, file_date))
        except OSError:
            continue
            
    if not files:
        return None
        
    # Сортируем по дате (самые новые в начале)
    files.sort(key=lambda x: x[1], reverse=True)
    
    # Загружаем самый свежий файл
    try:
        return load_json(files[0][0])
    except Exception as e:
        print(f"Ошибка при загрузке последних данных: {e}")
        return None


def merge_dataframes(dataframes: List[pd.DataFrame], 
                     sort_by: Optional[str] = 'date',
                     remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Объединяет несколько DataFrame в один.
    
    Аргументы:
        dataframes: Список DataFrame для объединения
        sort_by: Столбец для сортировки результата
        remove_duplicates: Удалять ли дубликаты
        
    Возвращает:
        Объединенный DataFrame
    """
    if not dataframes:
        raise APK.InvalidInputError("Список DataFrame для объединения пуст")
        
    # Объединяем все DataFrame
    merged = pd.concat(dataframes, ignore_index=True)
    
    # Удаляем дубликаты, если требуется
    if remove_duplicates and sort_by in merged.columns:
        merged.drop_duplicates(subset=[sort_by], keep='first', inplace=True)
        
    # Сортируем результат
    if sort_by in merged.columns:
        merged.sort_values(by=sort_by, inplace=True)
        
    return merged


# Инициализация модуля
build_file_index()


if __name__ == "__main__":
    try:
        # Пример использования функций модуля
        print(f"Общее количество файлов: {len(list_files())}")
        
        # Данные для тестового сохранения
        test_data = pd.DataFrame({
            'date': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'open': [100.0],
            'high': [105.0],
            'low': [98.0],
            'close': [102.0],
            'volume': [1000]
        })
        
        # Тестовое сохранение
        save_json(test_data, "TEST_DEMO_FILE.json", add_metadata=True, ticker="TEST")
        
        # Тестовая загрузка
        loaded_data = load_json("TEST_DEMO_FILE.json")
        print(f"Загруженные данные:\n{loaded_data}")
        
        # Удаление тестового файла
        delete_file("TEST_DEMO_FILE.json")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")