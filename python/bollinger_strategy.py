import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Optional, Union
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Константы для стратегии Боллинджера
DEFAULT_WINDOW = 20
DEFAULT_K = 2.0
MIN_K = 1.5  # Минимальный множитель для полос
MAX_K = 3.0  # Максимальный множитель для полос
MIN_STRENGTH = 0.5  # Минимальная сила сигнала
TREND_WINDOW = 5  # Окно для подтверждения тренда


def load_data(filename: str) -> pd.DataFrame:
    """
    Загружает данные из JSON-файла и конвертирует их в DataFrame.
    
    Аргументы:
        filename: Имя файла или полный путь к нему
        
    Возвращает:
        DataFrame с загруженными данными
        
    Вызывает:
        APK.DatabaseError: если файл не найден
        APK.InvalidInputError: при ошибке чтения или отсутствии необходимых столбцов
    """
    try:
        filepath = filename if os.path.dirname(filename) else os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            raise APK.DatabaseError(f"Файл данных не найден: {filename}")
        
        data = pd.read_json(filepath)
        
        # Проверка необходимых столбцов
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise APK.InvalidInputError(f"Некоторые обязательные столбцы отсутствуют: {', '.join(missing_columns)}")
        
        return data
        
    except pd.errors.ParserError as e:
        raise APK.InvalidInputError(f"Ошибка при чтении файла: {e}")
    except Exception as e:
        raise APK.DatabaseError(f"Неожиданная ошибка при загрузке данных: {e}")


def calculate_market_regime(data: pd.DataFrame, window: int = 20) -> str:
    """
    Определяет текущий режим рынка на основе анализа тренда и волатильности.
    
    Аргументы:
        data: DataFrame с данными
        window: Размер окна для анализа
        
    Возвращает:
        Строка с описанием режима рынка ('trending', 'ranging', 'volatile')
    """
    if len(data) < window * 2:
        return "unknown"  # Недостаточно данных
    
    # Получаем последние данные для анализа
    recent_data = data.tail(window * 2)
    
    # Рассчитываем тренд (наклон линейной регрессии)
    x = np.arange(len(recent_data))
    y = recent_data['close'].values
    slope, _ = np.polyfit(x, y, 1)
    
    # Волатильность (стандартное отклонение дневных изменений)
    recent_volatility = recent_data['close'].pct_change().std()
    
    # Средняя волатильность для всего набора данных
    avg_volatility = data['close'].pct_change().std()
    
    # Определяем режим рынка
    if recent_volatility > avg_volatility * 1.5:
        return "volatile"  # Волатильный рынок
    elif abs(slope) > 0.001:  # Пороговое значение наклона
        return "trending"  # Трендовый рынок
    else:
        return "ranging"  # Боковой рынок


def calculate_bollinger_bands(data: pd.DataFrame, window: int = DEFAULT_WINDOW, k: float = DEFAULT_K) -> pd.DataFrame:
    """
    Вычисляет Полосы Боллинджера с адаптивными параметрами.
    
    Аргументы:
        data: DataFrame с данными
        window: Размер окна для скользящего среднего
        k: Множитель для стандартного отклонения
        
    Возвращает:
        DataFrame с добавленными колонками для полос Боллинджера
    """
    # Копируем данные
    data = data.copy()
    
    # Определяем режим рынка
    market_regime = calculate_market_regime(data)
    
    # Расчет базовой волатильности
    volatility = data['close'].pct_change().std()
    
    # Адаптивные параметры на основе режима рынка и волатильности
    if market_regime == "volatile":
        adaptive_k = min(MAX_K, k * (1 + volatility * 2))
        adaptive_window = max(10, int(window * 0.8))  # Уменьшаем окно для быстрой реакции
    elif market_regime == "trending":
        adaptive_k = max(MIN_K, k * (1 + volatility))
        adaptive_window = window  # Стандартное окно
    else:  # "ranging" или "unknown"
        adaptive_k = min(MAX_K, max(MIN_K, k * (1 + volatility * 0.5)))
        adaptive_window = min(40, int(window * 1.2))  # Увеличиваем окно для снижения шума
    
    # Расчет базовых показателей
    data['SMA'] = data['close'].rolling(window=adaptive_window).mean()
    data['STD'] = data['close'].rolling(window=adaptive_window).std()
    
    # Расчет полос с адаптивными параметрами
    data['Upper'] = data['SMA'] + (adaptive_k * data['STD'])
    data['Lower'] = data['SMA'] - (adaptive_k * data['STD'])
    
    # Добавляем индикатор силы сигнала (расстояние от цены до SMA, нормализованное STD)
    data['Signal_Strength'] = abs((data['close'] - data['SMA']) / data['STD'])
    
    # Добавляем тренд
    data['Trend'] = data['close'].rolling(window=adaptive_window).mean().diff()
    
    # Добавляем ширину канала в процентах для оценки волатильности
    data['Band_Width'] = (data['Upper'] - data['Lower']) / data['SMA'] * 100
    
    # Добавляем индикатор перекупленности/перепроданности для более точных сигналов
    data['BB_Position'] = (data['close'] - data['Lower']) / (data['Upper'] - data['Lower'])
    
    return data


def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует торговые сигналы с фильтрацией ложных сигналов.
    
    Аргументы:
        data: DataFrame с рассчитанными показателями Боллинджера
        
    Возвращает:
        DataFrame с добавленными сигналами
    """
    # Копируем DataFrame
    data = data.copy()
    
    # Инициализация сигналов
    data['Signal'] = 0
    
    # Параметры фильтрации
    min_strength = MIN_STRENGTH  # Минимальная сила сигнала
    trend_window = TREND_WINDOW  # Окно для подтверждения тренда
    
    # Определяем режим рынка для адаптации стратегии
    market_regime = calculate_market_regime(data)
    
    # Адаптируем параметры к режиму рынка
    if market_regime == "volatile":
        min_strength = min_strength * 1.5  # Требуем более сильных сигналов в волатильном рынке
    elif market_regime == "ranging":
        min_strength = min_strength * 0.8  # Снижаем требования в боковом рынке
    
    # Сигналы на покупку с учетом режима рынка
    if market_regime != "volatile":  # В волатильном рынке более строгие условия
        buy_conditions = (
            (data['close'] < data['Lower']) &                     # Цена ниже нижней полосы
            (data['Signal_Strength'] > min_strength) &            # Достаточная сила сигнала
            (data['Trend'] > 0) &                                 # Положительный тренд
            (data['close'].rolling(trend_window).mean().diff() > 0)  # Подтверждение тренда
        )
    else:
        # В волатильном рынке добавляем проверку на разворот
        buy_conditions = (
            (data['close'] < data['Lower']) &                     # Цена ниже нижней полосы
            (data['Signal_Strength'] > min_strength) &            # Достаточная сила сигнала
            (data['Trend'] > 0) &                                 # Положительный тренд
            (data['close'].rolling(trend_window).mean().diff() > 0) &  # Подтверждение тренда
            (data['BB_Position'].shift(1) < data['BB_Position'])  # Движение от нижней полосы вверх
        )
    
    # Сигналы на продажу с аналогичной логикой
    if market_regime != "volatile":
        sell_conditions = (
            (data['close'] > data['Upper']) &                     # Цена выше верхней полосы
            (data['Signal_Strength'] > min_strength) &            # Достаточная сила сигнала
            (data['Trend'] < 0) &                                 # Отрицательный тренд
            (data['close'].rolling(trend_window).mean().diff() < 0)  # Подтверждение тренда
        )
    else:
        sell_conditions = (
            (data['close'] > data['Upper']) &                     # Цена выше верхней полосы
            (data['Signal_Strength'] > min_strength) &            # Достаточная сила сигнала
            (data['Trend'] < 0) &                                 # Отрицательный тренд
            (data['close'].rolling(trend_window).mean().diff() < 0) &  # Подтверждение тренда
            (data['BB_Position'].shift(1) > data['BB_Position'])  # Движение от верхней полосы вниз
        )
    
    # Установка сигналов
    data.loc[buy_conditions, 'Signal'] = 1    # Сигнал на покупку
    data.loc[sell_conditions, 'Signal'] = -1  # Сигнал на продажу
    
    # Добавляем силу сигнала на основе позиции в канале
    data.loc[data['Signal'] == 1, 'Signal_Power'] = (0.5 - data.loc[data['Signal'] == 1, 'BB_Position']) * 2
    data.loc[data['Signal'] == -1, 'Signal_Power'] = (data.loc[data['Signal'] == -1, 'BB_Position'] - 0.5) * 2
    data.loc[data['Signal'] == 0, 'Signal_Power'] = 0
    
    # Расчет позиции
    data['Position'] = data['Signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    return data


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет доходность стратегии с учетом силы сигналов.
    
    Аргументы:
        data: DataFrame с сигналами
        
    Возвращает:
        DataFrame с расчетом доходности
    """
    data = data.copy()
    
    # Рассчитываем доходность
    data['Returns'] = data['close'].pct_change()
    
    # Учитываем силу сигнала при расчете доходности стратегии
    if 'Signal_Power' in data.columns:
        data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1) * (1 + data['Signal_Power'].shift(1) * 0.2)
    else:
        data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1)
    
    # Кумулятивная доходность
    data['Cumulative_Returns'] = (1 + data['Strategy_Returns']).cumprod()
    
    # Добавляем расчет эффективности сигналов
    data['Signal_Effectiveness'] = data['Strategy_Returns'].rolling(window=5).mean()
    
    # Добавляем максимальную просадку
    data['Drawdown'] = 1 - data['Cumulative_Returns'] / data['Cumulative_Returns'].cummax()
    data['Max_Drawdown'] = data['Drawdown'].cummax()
    
    return data


def generate_recommendation(data: pd.DataFrame) -> Dict[str, any]:
    """
    Анализирует последние данные и выдает детальную рекомендацию.
    
    Аргументы:
        data: DataFrame с рассчитанными показателями
        
    Возвращает:
        Словарь с рекомендацией и дополнительной информацией
    """
    # Проверяем наличие необходимых данных
    if data.empty or 'close' not in data.columns or 'SMA' not in data.columns:
        return {
            "action": "НЕИЗВЕСТНО",
            "confidence": "низкая",
            "reason": "Недостаточно данных для анализа",
            "description": "Невозможно сформировать рекомендацию из-за отсутствия необходимых данных."
        }
    
    # Получаем последние данные
    latest_data = data.iloc[-1]
    
    # Определяем режим рынка
    market_regime = calculate_market_regime(data)
    
    # Базовые условия
    price_vs_bands = (
        "выше верхней полосы" if latest_data['close'] > latest_data['Upper']
        else "ниже нижней полосы" if latest_data['close'] < latest_data['Lower']
        else "внутри полос"
    )
    
    # Анализ тренда
    trend_strength = "сильный" if 'Signal_Strength' in latest_data and latest_data['Signal_Strength'] > 1 else "слабый"
    trend_direction = "восходящий" if 'Trend' in latest_data and latest_data['Trend'] > 0 else "нисходящий"
    
    # Анализ волатильности
    if 'Band_Width' in latest_data:
        volatility = (
            "высокая" if latest_data['Band_Width'] > data['Band_Width'].mean() * 1.5
            else "низкая" if latest_data['Band_Width'] < data['Band_Width'].mean() * 0.5
            else "средняя"
        )
    else:
        volatility = "неизвестная"
    
    # Формирование рекомендации
    recommendation = {}
    
    if latest_data['close'] < latest_data['Lower'] and ('Trend' not in latest_data or latest_data['Trend'] > 0):
        recommendation["action"] = "ПОКУПАТЬ"
        confidence = "высокая" if 'Signal_Strength' in latest_data and latest_data['Signal_Strength'] > 1 else "средняя"
        reason = f"Цена {price_vs_bands}, {trend_direction} тренд {trend_strength}"
        
        # Модифицируем рекомендацию на основе режима рынка
        if market_regime == "volatile":
            recommendation["action"] = "ПОКУПАТЬ С ОСТОРОЖНОСТЬЮ"
            confidence = "средняя"
            reason += f", но высокая волатильность рынка ({volatility})"
    elif latest_data['close'] > latest_data['Upper'] and ('Trend' not in latest_data or latest_data['Trend'] < 0):
        recommendation["action"] = "ПРОДАВАТЬ"
        confidence = "высокая" if 'Signal_Strength' in latest_data and latest_data['Signal_Strength'] > 1 else "средняя"
        reason = f"Цена {price_vs_bands}, {trend_direction} тренд {trend_strength}"
        
        # Модифицируем рекомендацию на основе режима рынка
        if market_regime == "volatile":
            recommendation["action"] = "ПРОДАВАТЬ С ОСТОРОЖНОСТЬЮ"
            confidence = "средняя"
            reason += f", но высокая волатильность рынка ({volatility})"
    else:
        recommendation["action"] = "УДЕРЖИВАТЬ"
        confidence = "средняя"
        reason = f"Цена {price_vs_bands}, недостаточно сильных сигналов"
    
    # Заполняем оставшиеся поля рекомендации
    recommendation["confidence"] = confidence
    recommendation["reason"] = reason
    recommendation["market_regime"] = market_regime
    recommendation["volatility"] = volatility
    recommendation["trend_direction"] = trend_direction
    recommendation["trend_strength"] = trend_strength
    
    # Добавляем текстовое описание
    description = (
        f"Рекомендация: {recommendation['action']}\n"
        f"Уверенность: {recommendation['confidence']}\n"
        f"Причина: {recommendation['reason']}\n"
        f"Режим рынка: {recommendation['market_regime']}\n"
        f"Волатильность: {recommendation['volatility']}\n"
    )
    
    if 'Signal_Strength' in latest_data:
        description += f"Сила сигнала: {latest_data['Signal_Strength']:.2f}\n"
    
    recommendation["description"] = description
    
    return recommendation


def bollinger_strings(dataFrame: pd.DataFrame) -> APK.bollinger:
    """
    Выполняет анализ полос Боллинджера и возвращает объект с результатами.
    
    Аргументы:
        dataFrame: DataFrame с исходными данными
        
    Возвращает:
        Объект класса APK.bollinger с результатами анализа
    """
    try:
        # Загрузка данных
        data = dataFrame.copy()
        
        # Расчет индикаторов и генерация сигналов
        data = calculate_bollinger_bands(data)
        data = generate_signals(data)
        data = calculate_returns(data)
        
        # Получение рекомендации
        recommendation = generate_recommendation(data)
        
        # Подготовка данных для возврата
        columns_to_keep = ["Upper", "Lower", 'Signal']
        
        # Создаем объект bollinger
        bollinger_class = APK.bollinger(recommendation["description"], data[columns_to_keep])
        
        return bollinger_class
    except APK.ApplicationError as e:
        print(f"Ошибка при анализе Боллинджера: {e}")
        return False
    except Exception as e:
        print(f"Непредвиденная ошибка при анализе Боллинджера: {e}")
        return False


def plot_bollinger_bands(data: pd.DataFrame, output_file: Optional[str] = None) -> None:
    """
    Создает график полос Боллинджера.
    
    Аргументы:
        data: DataFrame с рассчитанными показателями Боллинджера
        output_file: Путь для сохранения графика (опционально)
    """
    plt.figure(figsize=(12, 6))
    
    # Строим график цены и полос Боллинджера
    plt.plot(data.index, data['close'], label='Цена закрытия', color='blue')
    plt.plot(data.index, data['SMA'], label='SMA', color='red', alpha=0.6)
    plt.plot(data.index, data['Upper'], label='Верхняя полоса', color='green', linestyle='--', alpha=0.5)
    plt.plot(data.index, data['Lower'], label='Нижняя полоса', color='green', linestyle='--', alpha=0.5)
    
    # Отмечаем сигналы на покупку и продажу
    buy_signals = data[data['Signal'] == 1].index
    sell_signals = data[data['Signal'] == -1].index
    
    plt.scatter(buy_signals, data.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Сигнал покупки')
    plt.scatter(sell_signals, data.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Сигнал продажи')
    
    # Настраиваем график
    plt.title('Анализ на основе полос Боллинджера')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Сохраняем или показываем график
    if output_file:
        plt.savefig(output_file)
        print(f"График сохранен: {output_file}")
    else:
        plt.show()


def plot_strategy_performance(data: pd.DataFrame, output_file: Optional[str] = None) -> None:
    """
    Создает график эффективности стратегии.
    
    Аргументы:
        data: DataFrame с рассчитанными показателями доходности
        output_file: Путь для сохранения графика (опционально)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # График кумулятивной доходности
    ax1.plot(data.index, data['Cumulative_Returns'], label='Доходность стратегии', color='blue')
    ax1.axhline(y=1, color='r', linestyle='--', alpha=0.3)
    ax1.set_title('Эффективность стратегии')
    ax1.set_ylabel('Кумулятивная доходность')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График просадки
    ax2.fill_between(data.index, 0, data['Drawdown'], color='red', alpha=0.3)
    ax2.set_title('Просадка')
    ax2.set_ylabel('Просадка')
    ax2.set_xlabel('Дата')
    ax2.grid(True, alpha=0.3)
    
    # Настраиваем макет
    plt.tight_layout()
    
    # Сохраняем или показываем график
    if output_file:
        plt.savefig(output_file)
        print(f"График эффективности сохранен: {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    try:
        # Тестирование стратегии
        filename = os.path.join(DATA_DIR, "SBER_2025-01-26_9D_[171624].json")
        
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден, используем тестовые данные")
            
            # Создаем тестовые данные для демонстрации
            import numpy as np
            dates = pd.date_range(start='2023-01-01', periods=200)
            np.random.seed(42)
            
            # Генерируем случайные цены с трендом и циклами
            trend = np.linspace(0, 20, 200)
            cycle = 10 * np.sin(np.linspace(0, 15, 200))
            noise = np.random.normal(0, 5, 200)
            
            close_prices = 100 + trend + cycle + noise
            high_prices = close_prices + np.random.uniform(0, 5, 200)
            low_prices = close_prices - np.random.uniform(0, 5, 200)
            
            # Создаем DataFrame
            test_data = pd.DataFrame({
                'date': dates,
                'close': close_prices,
                'high': high_prices,
                'low': low_prices,
                'volume': np.random.randint(1000, 10000, 200)
            })
            
            # Сохраняем тестовые данные
            test_filename = os.path.join(DATA_DIR, "test_bollinger_data.json")
            test_data.to_json(test_filename)
            filename = test_filename
        
        # Загружаем данные
        data = load_data(filename)
        
        # Анализируем данные
        result = bollinger_strings(data)
        if result:
            print(result.recommendation)
            
            # Полный анализ данных
            full_data = calculate_bollinger_bands(data)
            full_data = generate_signals(full_data)
            full_data = calculate_returns(full_data)
            
            # Создаем и сохраняем графики
            plots_dir = os.path.join(DATA_DIR, "plots")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
                
            plot_bollinger_bands(full_data, os.path.join(plots_dir, "bollinger_bands.png"))
            plot_strategy_performance(full_data, os.path.join(plots_dir, "strategy_performance.png"))
        else:
            print("Ошибка расчета")
            
    except APK.ApplicationError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {e}")