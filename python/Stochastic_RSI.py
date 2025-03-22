import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Union
import _AppProjectKit as APK
from rsi_call import current_rsi_call, calculate_rsi

# Директория для хранения данных
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Константы для StochRSI
DEFAULT_RSI_PERIOD = 14      # Период для расчета RSI
DEFAULT_STOCH_PERIOD = 14    # Период для Стохастика
DEFAULT_K_PERIOD = 3         # Период сглаживания для %K
DEFAULT_D_PERIOD = 3         # Период сглаживания для %D
OVERBOUGHT_LEVEL = 80        # Уровень перекупленности
OVERSOLD_LEVEL = 20          # Уровень перепроданности


def calculate_stochastic_rsi(dataFrame: pd.DataFrame, 
                           rsi_period: int = DEFAULT_RSI_PERIOD, 
                           stoch_period: int = DEFAULT_STOCH_PERIOD,
                           smooth_k: int = DEFAULT_K_PERIOD, 
                           smooth_d: int = DEFAULT_D_PERIOD) -> pd.DataFrame:
    """
    Рассчитывает Стохастический RSI (StochRSI).
    
    Аргументы:
        dataFrame: DataFrame с ценовыми данными
        rsi_period: Период для расчёта RSI
        stoch_period: Период для Стохастического осциллятора
        smooth_k: Период сглаживания для %K линии
        smooth_d: Период сглаживания для %D линии
    
    Возвращает:
        DataFrame с добавленными колонками:
        - RSI: значения RSI
        - StochRSI_K: быстрая линия StochRSI
        - StochRSI_D: медленная линия StochRSI
        - StochRSI_Signal: сигналы (-1, 0, 1)
    """
    try:
        # Проверка входных данных
        if 'close' not in dataFrame.columns:
            raise APK.InvalidInputError("DataFrame должен содержать колонку 'close'")
            
        # Копируем DataFrame
        df = dataFrame.copy()
        
        # Получаем значения RSI
        df['RSI'] = current_rsi_call(df, rsi_period)
        
        # Рассчитываем Стохастический RSI
        rsi_series = df['RSI'].copy()
        
        # Очищаем от NaN значений для корректного расчета
        rsi_series = rsi_series.dropna()
        
        if len(rsi_series) < stoch_period + 1:
            raise APK.InvalidInputError(f"Недостаточно данных для расчета StochRSI. Требуется минимум {stoch_period + 1} значений RSI")
        
        # Создаем Series для StochRSI с тем же индексом, что и исходные данные
        stoch_rsi_k = pd.Series(index=df.index)
        stoch_rsi_d = pd.Series(index=df.index)
        
        # Для каждой точки вычисляем StochRSI
        for i in range(stoch_period - 1, len(rsi_series)):
            # Получаем выборку RSI для текущего периода
            window = rsi_series.iloc[i - stoch_period + 1:i + 1]
            
            # Находим минимум и максимум в окне
            min_rsi = window.min()
            max_rsi = window.max()
            
            # Избегаем деления на ноль
            if max_rsi == min_rsi:
                stoch_value = 50  # Нейтральное значение
            else:
                # Формула Стохастического осциллятора
                stoch_value = 100 * (rsi_series.iloc[i] - min_rsi) / (max_rsi - min_rsi)
                
            stoch_rsi_k.iloc[i] = stoch_value
        
        # Сглаживаем %K
        if smooth_k > 1:
            stoch_rsi_k = stoch_rsi_k.rolling(window=smooth_k).mean()
        
        # Рассчитываем %D (скользящее среднее %K)
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
        
        # Добавляем значения в DataFrame
        df['StochRSI_K'] = stoch_rsi_k
        df['StochRSI_D'] = stoch_rsi_d
        
        # Генерируем сигналы
        df['StochRSI_Signal'] = 0
        
        # Сигнал на покупку: StochRSI_K пересекает StochRSI_D снизу вверх при значениях ниже 20
        buy_condition = (
            (df['StochRSI_K'] > df['StochRSI_D']) & 
            (df['StochRSI_K'].shift(1) <= df['StochRSI_D'].shift(1)) &
            (df['StochRSI_K'] < OVERSOLD_LEVEL)
        )
        df.loc[buy_condition, 'StochRSI_Signal'] = 1
        
        # Сигнал на продажу: StochRSI_K пересекает StochRSI_D сверху вниз при значениях выше 80
        sell_condition = (
            (df['StochRSI_K'] < df['StochRSI_D']) & 
            (df['StochRSI_K'].shift(1) >= df['StochRSI_D'].shift(1)) &
            (df['StochRSI_K'] > OVERBOUGHT_LEVEL)
        )
        df.loc[sell_condition, 'StochRSI_Signal'] = -1
        
        # Расчет силы сигнала на основе разницы между K и D
        df['StochRSI_Signal_Strength'] = abs(df['StochRSI_K'] - df['StochRSI_D']) / 10
        
        return df
        
    except Exception as e:
        raise APK.ApplicationError(f"Ошибка при расчете Stochastic RSI: {str(e)}")


def refine_stoch_signals(df: pd.DataFrame, price_trend_window: int = 10) -> pd.DataFrame:
    """
    Улучшает сигналы StochRSI с учетом тренда цены.
    
    Аргументы:
        df: DataFrame с рассчитанным StochRSI
        price_trend_window: Окно для определения тренда цены
        
    Возвращает:
        DataFrame с уточненными сигналами
    """
    result = df.copy()
    
    # Определяем тренд цены (по простой линейной регрессии)
    result['Price_Trend'] = 0
    
    for i in range(price_trend_window, len(result)):
        # Получаем окно цен
        prices = result['close'].iloc[i-price_trend_window:i]
        
        # Линейная регрессия
        x = np.arange(len(prices))
        y = prices.values
        slope, _ = np.polyfit(x, y, 1)
        
        # Определяем тренд на основе наклона
        if slope > 0:
            result.loc[result.index[i], 'Price_Trend'] = 1  # Восходящий
        elif slope < 0:
            result.loc[result.index[i], 'Price_Trend'] = -1  # Нисходящий
    
    # Корректируем сигналы с учетом тренда
    # Усиливаем сигналы, совпадающие с направлением тренда
    result['StochRSI_Signal_Refined'] = result['StochRSI_Signal']
    
    # Если сигнал и тренд совпадают, усиливаем сигнал
    result.loc[
        (result['StochRSI_Signal'] == 1) & (result['Price_Trend'] == 1), 
        'StochRSI_Signal_Refined'
    ] = 2
    
    result.loc[
        (result['StochRSI_Signal'] == -1) & (result['Price_Trend'] == -1), 
        'StochRSI_Signal_Refined'
    ] = -2
    
    # Если сигнал противоречит сильному тренду, ослабляем его
    result.loc[
        (result['StochRSI_Signal'] == 1) & (result['Price_Trend'] == -1), 
        'StochRSI_Signal_Refined'
    ] = 0.5
    
    result.loc[
        (result['StochRSI_Signal'] == -1) & (result['Price_Trend'] == 1), 
        'StochRSI_Signal_Refined'
    ] = -0.5
    
    return result


def detect_stoch_rsi_divergences(df: pd.DataFrame, window: int = 20) -> List[Dict[str, Union[str, int, float]]]:
    """
    Обнаруживает дивергенции между StochRSI и ценой.
    
    Аргументы:
        df: DataFrame с рассчитанным StochRSI
        window: Окно для поиска локальных экстремумов
        
    Возвращает:
        Список обнаруженных дивергенций
    """
    # Проверка наличия необходимых данных
    if 'StochRSI_K' not in df.columns or 'close' not in df.columns:
        return []
    
    # Очищаем от NaN
    clean_df = df.dropna(subset=['StochRSI_K', 'close'])
    
    if len(clean_df) < window * 2:
        return []
    
    divergences = []
    
    # Ищем локальные максимумы и минимумы для StochRSI_K
    stoch_peaks = []
    stoch_troughs = []
    
    for i in range(window, len(clean_df) - window):
        idx = clean_df.index[i]
        
        # Локальный максимум StochRSI
        if all(clean_df['StochRSI_K'].iloc[i] > clean_df['StochRSI_K'].iloc[i-j] for j in range(1, window+1)) and \
           all(clean_df['StochRSI_K'].iloc[i] > clean_df['StochRSI_K'].iloc[i+j] for j in range(1, window+1)):
            stoch_peaks.append(i)
        
        # Локальный минимум StochRSI
        if all(clean_df['StochRSI_K'].iloc[i] < clean_df['StochRSI_K'].iloc[i-j] for j in range(1, window+1)) and \
           all(clean_df['StochRSI_K'].iloc[i] < clean_df['StochRSI_K'].iloc[i+j] for j in range(1, window+1)):
            stoch_troughs.append(i)
    
    # Ищем дивергенции в максимумах (медвежьи)
    for i in range(len(stoch_peaks) - 1):
        if stoch_peaks[i+1] - stoch_peaks[i] > window:  # Убедимся, что пики достаточно далеко друг от друга
            idx1 = clean_df.index[stoch_peaks[i]]
            idx2 = clean_df.index[stoch_peaks[i+1]]
            
            # Медвежья дивергенция: цена делает более высокий максимум, но StochRSI делает более низкий
            if (clean_df.loc[idx2, 'close'] > clean_df.loc[idx1, 'close'] and 
                clean_df.loc[idx2, 'StochRSI_K'] < clean_df.loc[idx1, 'StochRSI_K']):
                
                divergences.append({
                    "type": "bearish",
                    "indicator": "StochRSI",
                    "position": stoch_peaks[i+1],
                    "bars_ago": len(clean_df) - 1 - stoch_peaks[i+1],
                    "strength": min(10, abs(clean_df.loc[idx2, 'StochRSI_K'] - clean_df.loc[idx1, 'StochRSI_K'])) * 0.5
                })
    
    # Ищем дивергенции в минимумах (бычьи)
    for i in range(len(stoch_troughs) - 1):
        if stoch_troughs[i+1] - stoch_troughs[i] > window:  # Убедимся, что впадины достаточно далеко друг от друга
            idx1 = clean_df.index[stoch_troughs[i]]
            idx2 = clean_df.index[stoch_troughs[i+1]]
            
            # Бычья дивергенция: цена делает более низкий минимум, но StochRSI делает более высокий
            if (clean_df.loc[idx2, 'close'] < clean_df.loc[idx1, 'close'] and 
                clean_df.loc[idx2, 'StochRSI_K'] > clean_df.loc[idx1, 'StochRSI_K']):
                
                divergences.append({
                    "type": "bullish",
                    "indicator": "StochRSI",
                    "position": stoch_troughs[i+1],
                    "bars_ago": len(clean_df) - 1 - stoch_troughs[i+1],
                    "strength": min(10, abs(clean_df.loc[idx2, 'StochRSI_K'] - clean_df.loc[idx1, 'StochRSI_K'])) * 0.5
                })
    
    return divergences


def plot_stochastic_rsi(df: pd.DataFrame, title: str = "Stochastic RSI Analysis", 
                      save_path: Optional[str] = None) -> None:
    """
    Создает график StochRSI, RSI и цены.
    
    Аргументы:
        df: DataFrame с рассчитанными показателями
        title: Заголовок графика
        save_path: Путь для сохранения графика (если None, график отображается)
    """
    # Проверка наличия необходимых данных
    if not all(col in df.columns for col in ['close', 'RSI', 'StochRSI_K', 'StochRSI_D']):
        print("Ошибка: отсутствуют необходимые данные для построения графика")
        return
    
    # Очищаем данные от NaN
    clean_df = df.dropna(subset=['StochRSI_K', 'StochRSI_D'])
    
    if len(clean_df) < 2:
        print("Недостаточно данных для построения графика")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # График цены
    ax1.plot(clean_df.index, clean_df['close'])
    ax1.set_title(title)
    ax1.set_ylabel("Цена")
    ax1.grid(True, alpha=0.3)
    
    # График RSI
    ax2.plot(clean_df.index, clean_df['RSI'], color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='k', linestyle='-', alpha=0.2)
    ax2.fill_between(clean_df.index, 70, 100, color='red', alpha=0.1)
    ax2.fill_between(clean_df.index, 0, 30, color='green', alpha=0.1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(True, alpha=0.3)
    
    # График StochRSI
    ax3.plot(clean_df.index, clean_df['StochRSI_K'], color='blue', label='%K')
    ax3.plot(clean_df.index, clean_df['StochRSI_D'], color='red', label='%D')
    ax3.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=20, color='g', linestyle='--', alpha=0.5)
    ax3.fill_between(clean_df.index, 80, 100, color='red', alpha=0.1)
    ax3.fill_between(clean_df.index, 0, 20, color='green', alpha=0.1)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("StochRSI")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Отмечаем сигналы, если они есть
    if 'StochRSI_Signal' in clean_df.columns:
        buy_signals = clean_df[clean_df['StochRSI_Signal'] == 1].index
        sell_signals = clean_df[clean_df['StochRSI_Signal'] == -1].index
        
        for idx in buy_signals:
            ax3.scatter(idx, clean_df.loc[idx, 'StochRSI_K'], marker='^', color='green', s=100)
            ax1.scatter(idx, clean_df.loc[idx, 'close'], marker='^', color='green', s=100)
        
        for idx in sell_signals:
            ax3.scatter(idx, clean_df.loc[idx, 'StochRSI_K'], marker='v', color='red', s=100)
            ax1.scatter(idx, clean_df.loc[idx, 'close'], marker='v', color='red', s=100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_stoch_rsi_summary(data: pd.DataFrame) -> Dict[str, Union[str, float, int]]:
    """
    Создает сводку по текущим значениям Stochastic RSI
    
    Аргументы:
        data: DataFrame с рассчитанными показателями StochRSI
        
    Возвращает:
        Словарь с текущим состоянием индикатора
    """
    try:
        # Проверяем наличие необходимых данных
        required_columns = ['RSI', 'StochRSI_K', 'StochRSI_D']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            # Если данные не рассчитаны, делаем это
            if 'RSI' not in data.columns and 'close' in data.columns:
                data = calculate_stochastic_rsi(data)
            else:
                raise APK.InvalidInputError(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
        
        # Получаем последние данные
        latest = data.iloc[-1]
        
        # Определяем состояние
        if latest['StochRSI_K'] > OVERBOUGHT_LEVEL:
            condition = "Перекуплен"
        elif latest['StochRSI_K'] < OVERSOLD_LEVEL:
            condition = "Перепродан"
        else:
            condition = "Нейтральный"
            
        # Определяем тренд
        if latest['StochRSI_K'] > latest['StochRSI_D']:
            trend = "Восходящий"
        elif latest['StochRSI_K'] < latest['StochRSI_D']:
            trend = "Нисходящий"
        else:
            trend = "Боковой"
        
        # Формируем рекомендацию
        if condition == "Перепродан" and trend == "Восходящий":
            recommendation = "Покупка"
            signal_strength = 8 if latest['StochRSI_K'] < 10 else 6
        elif condition == "Перекуплен" and trend == "Нисходящий":
            recommendation = "Продажа"
            signal_strength = 8 if latest['StochRSI_K'] > 90 else 6
        elif trend == "Восходящий" and latest['StochRSI_K'] < 50:
            recommendation = "Слабая покупка"
            signal_strength = 4
        elif trend == "Нисходящий" and latest['StochRSI_K'] > 50:
            recommendation = "Слабая продажа"
            signal_strength = 4
        else:
            recommendation = "Нейтрально"
            signal_strength = 2
            
        # Проверяем наличие дивергенций
        divergences = []
        
        if len(data) > 40:  # Минимум данных для анализа дивергенций
            divergences = detect_stoch_rsi_divergences(data)
            
            # Корректируем рекомендацию на основе свежих дивергенций
            recent_divergences = [d for d in divergences if d['bars_ago'] < 5]
            for div in recent_divergences:
                if div['type'] == 'bullish' and recommendation in ["Нейтрально", "Слабая покупка"]:
                    recommendation = "Покупка (дивергенция)"
                    signal_strength = max(signal_strength, 7)
                elif div['type'] == 'bearish' and recommendation in ["Нейтрально", "Слабая продажа"]:
                    recommendation = "Продажа (дивергенция)"
                    signal_strength = max(signal_strength, 7)
        
        # Формируем итоговую сводку
        summary = {
            'condition': condition,
            'trend': trend,
            'rsi': latest['RSI'],
            'stoch_k': latest['StochRSI_K'],
            'stoch_d': latest['StochRSI_D'],
            'signal': latest['StochRSI_Signal'] if 'StochRSI_Signal' in latest else 0,
            'recommendation': recommendation,
            'signal_strength': signal_strength,
            'divergences': divergences
        }
        
        return summary
        
    except Exception as e:
        raise APK.ApplicationError(f"Ошибка создания сводки Stochastic RSI: {str(e)}")


if __name__ == "__main__":
    try:
        # Поиск файла с данными
        test_file = os.path.join(DATA_DIR, "FEES_2024-11-10_3D_[183522].json")
        
        if not os.path.exists(test_file):
            # Ищем любой JSON-файл в директории
            json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            if json_files:
                test_file = os.path.join(DATA_DIR, json_files[0])
                print(f"Используем файл: {test_file}")
            else:
                # Создаем тестовые данные
                print("Файлы не найдены. Создаем тестовые данные...")
                import numpy as np
                
                dates = pd.date_range(start='2023-01-01', periods=200)
                close_prices = []
                
                # Генерируем цены с трендами и циклами
                price = 100.0
                for i in range(200):
                    # Добавляем циклы для демонстрации StochRSI
                    cycle = 5 * np.sin(i / 10)
                    # Добавляем тренд
                    if i < 50:
                        trend = 0.1  # Восходящий
                    elif i < 100:
                        trend = -0.15  # Нисходящий
                    elif i < 150:
                        trend = 0.05  # Слабо восходящий
                    else:
                        trend = 0  # Боковой
                    
                    # Добавляем случайность
                    noise = np.random.normal(0, 1)
                    
                    # Обновляем цену
                    price += trend + cycle + noise
                    close_prices.append(max(price, 50))  # Не допускаем падения ниже 50
                
                test_data = pd.DataFrame({
                    'close': close_prices,
                    'high': [p + abs(np.random.normal(0, 0.5)) for p in close_prices],
                    'low': [p - abs(np.random.normal(0, 0.5)) for p in close_prices],
                    'volume': np.random.randint(1000, 5000, 200)
                }, index=dates)
                
                # Сохраняем тестовые данные
                test_file = os.path.join(DATA_DIR, "test_stochrsi_data.json")
                test_data.to_json(test_file)
                print(f"Тестовые данные сохранены в {test_file}")
        
        # Загрузка данных
        data = pd.read_json(test_file)
        
        # Расчет Stochastic RSI
        result = calculate_stochastic_rsi(data)
        
        # Уточнение сигналов с учетом тренда цены
        result = refine_stoch_signals(result)
        
        # Получение сводки
        summary = get_stoch_rsi_summary(result)
        
        # Вывод результатов
        print("\nПоследние 10 значений Stochastic RSI:")
        print(result[['RSI', 'StochRSI_K', 'StochRSI_D', 'StochRSI_Signal']].tail(10))
        
        print("\nТекущее состояние:")
        print(f"Состояние: {summary['condition']}")
        print(f"Тренд: {summary['trend']}")
        print(f"RSI: {summary['rsi']:.2f}")
        print(f"Стохастик %K: {summary['stoch_k']:.2f}")
        print(f"Стохастик %D: {summary['stoch_d']:.2f}")
        print(f"Сигнал: {summary['signal']}")
        print(f"Рекомендация: {summary['recommendation']}")
        print(f"Сила сигнала: {summary['signal_strength']}/10")
        
        # Вывод информации о дивергенциях
        if summary['divergences']:
            print("\nОбнаруженные дивергенции:")
            for div in summary['divergences']:
                print(f"- {div['type'].capitalize()} дивергенция, {div['bars_ago']} баров назад, сила: {div['strength']:.1f}")
        
        # Построение графика
        plots_dir = os.path.join(DATA_DIR, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        plot_stochastic_rsi(result, save_path=os.path.join(plots_dir, "stoch_rsi.png"))
        print(f"\nГрафик сохранен в {plots_dir}/stoch_rsi.png")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")