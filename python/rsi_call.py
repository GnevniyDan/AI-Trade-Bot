import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional, Union
import _AppProjectKit as APK

# Директория для хранения данных
DATA_DIR = APK.DATA_DIR
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Константы для RSI
DEFAULT_PERIOD = 14       # Стандартный период RSI
OVERBOUGHT_LEVEL = 70     # Уровень перекупленности
OVERSOLD_LEVEL = 30       # Уровень перепроданности
NEUTRAL_ZONE = (45, 55)   # Нейтральная зона
SIGNAL_THRESHOLD = 3      # Минимальное количество баров для подтверждения сигнала


def calculate_rsi(data: pd.Series, period: int = DEFAULT_PERIOD) -> pd.Series:
    """
    Рассчитывает индекс относительной силы (RSI) для временного ряда.
    
    Аргументы:
        data: Временной ряд цен (pd.Series)
        period: Период для расчета RSI
        
    Возвращает:
        Series с рассчитанным RSI
    """
    # Проверка длины серии
    if len(data) < period + 1:
        raise APK.DatabaseError(f"Недостаточно данных для расчета RSI с периодом {period}. Требуется минимум {period + 1}, имеется {len(data)}.")
    
    # Расчет изменений цены
    delta = data.diff()
    
    # Разделение на положительные и отрицательные изменения
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Инициализация первого значения
    avg_gain = gain.rolling(window=period, min_periods=1).mean().iloc[period-1]
    avg_loss = loss.rolling(window=period, min_periods=1).mean().iloc[period-1]
    
    # Используем метод сглаживания Wilder's
    rsi_values = [np.nan] * period
    
    for i in range(period, len(data)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period
        
        if avg_loss == 0:
            rsi = 100  # Избегаем деления на ноль
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        rsi_values.append(rsi)
    
    return pd.Series(rsi_values, index=data.index)


def adaptive_rsi(data: pd.Series, base_period: int = DEFAULT_PERIOD, 
                volatility_factor: float = 0.5) -> pd.Series:
    """
    Рассчитывает адаптивный RSI, который меняет период в зависимости от волатильности.
    
    Аргументы:
        data: Временной ряд цен
        base_period: Базовый период RSI
        volatility_factor: Фактор влияния волатильности на период RSI
        
    Возвращает:
        Series с адаптивным RSI
    """
    # Расчет волатильности (стандартное отклонение процентных изменений)
    pct_change = data.pct_change()
    volatility = pct_change.rolling(window=base_period).std()
    
    # Нормализация волатильности (относительно среднего значения за весь период)
    avg_volatility = volatility.mean()
    normalized_vol = volatility / avg_volatility
    
    # Адаптивные периоды - уменьшаем в периоды высокой волатильности
    # и увеличиваем в периоды низкой волатильности
    adaptive_periods = []
    
    for i in range(len(data)):
        if i < base_period or np.isnan(normalized_vol.iloc[i]):
            adaptive_periods.append(base_period)
        else:
            # Инвертируем влияние волатильности (выше волатильность - меньше период)
            period_adjust = 1 / (normalized_vol.iloc[i] ** volatility_factor)
            # Ограничиваем изменение периода от 0.5 до 2 от базового
            adj_period = base_period * min(max(period_adjust, 0.5), 2)
            adaptive_periods.append(round(adj_period))
    
    # Рассчитываем RSI с адаптивными периодами
    # Это приблизительный метод, для полного адаптивного RSI требуется более сложная реализация
    return calculate_rsi(data, round(sum(adaptive_periods) / len(adaptive_periods)))


def current_rsi_call(dataFrame: pd.DataFrame, candle_frame: int = DEFAULT_PERIOD) -> pd.Series:
    """
    Рассчитывает индекс относительной силы (RSI) для переданного DataFrame.

    Аргументы:
        dataFrame: pandas DataFrame с данными, содержащими колонку 'close'
        candle_frame: период для RSI (по умолчанию 14)

    Возвращает:
        pandas.Series с рассчитанным RSI
    """
    # Проверка наличия данных
    if 'close' not in dataFrame.columns:
        raise APK.InvalidInputError("DataFrame должен содержать колонку 'close'")
        
    # Колонка с закрытиями
    close = dataFrame['close']

    # Проверка длины серии
    if close.size < candle_frame + 1:
        raise APK.DatabaseError(f"DataFrame слишком короткий: {close.size} строк, требуется минимум {candle_frame + 1}.")

    # Расчет RSI
    return calculate_rsi(close, candle_frame)


def get_rsi_signals(rsi_values: pd.Series) -> Dict[str, Union[str, float, List[int]]]:
    """
    Анализирует RSI и возвращает торговые сигналы и состояние.
    
    Аргументы:
        rsi_values: Series со значениями RSI
        
    Возвращает:
        Словарь с сигналами и оценкой состояния
    """
    # Проверка наличия достаточного количества значений
    if len(rsi_values.dropna()) < 5:
        return {
            "state": "unknown",
            "signal": "neutral",
            "strength": 0,
            "rsi": float("nan"),
            "trend": "unknown"
        }
    
    # Последнее значение RSI
    current_rsi = rsi_values.iloc[-1]
    
    # Состояние RSI
    if current_rsi > OVERBOUGHT_LEVEL:
        state = "overbought"
    elif current_rsi < OVERSOLD_LEVEL:
        state = "oversold"
    elif NEUTRAL_ZONE[0] <= current_rsi <= NEUTRAL_ZONE[1]:
        state = "neutral"
    elif current_rsi < NEUTRAL_ZONE[0]:
        state = "weakening"
    else:  # current_rsi > NEUTRAL_ZONE[1]
        state = "strengthening"
    
    # Анализируем тренд RSI (последние 5 значений)
    recent_rsi = rsi_values.tail(5).dropna()
    
    if len(recent_rsi) < 3:
        trend = "unknown"
    else:
        # Используем линейную регрессию для определения тренда
        x = np.arange(len(recent_rsi))
        y = recent_rsi.values
        slope, _ = np.polyfit(x, y, 1)
        
        if slope > 0.5:
            trend = "strongly rising"
        elif slope > 0.1:
            trend = "rising"
        elif slope < -0.5:
            trend = "strongly falling"
        elif slope < -0.1:
            trend = "falling"
        else:
            trend = "flat"
    
    # Определяем сигналы и их силу
    signal = "neutral"
    strength = 0
    
    # Бычий сигнал (RSI выходит из зоны перепроданности)
    if state == "oversold" and trend in ["rising", "strongly rising"]:
        signal = "buy"
        strength = 2 if trend == "strongly rising" else 1
    
    # Медвежий сигнал (RSI выходит из зоны перекупленности)
    elif state == "overbought" and trend in ["falling", "strongly falling"]:
        signal = "sell"
        strength = 2 if trend == "strongly falling" else 1
    
    # Проверка расхождений (дивергенций)
    divergences = detect_rsi_divergences(rsi_values, pd.Series(rsi_values.index.values))
    
    # Возвращаем результаты
    return {
        "state": state,
        "signal": signal,
        "strength": strength,
        "rsi": current_rsi,
        "trend": trend,
        "divergences": divergences
    }


def detect_rsi_divergences(rsi: pd.Series, prices: pd.Series, 
                          window: int = 10) -> List[Dict[str, Union[str, int]]]:
    """
    Обнаруживает дивергенции между RSI и ценой.
    
    Аргументы:
        rsi: Series со значениями RSI
        prices: Series с ценами
        window: Окно для поиска локальных максимумов/минимумов
        
    Возвращает:
        Список обнаруженных дивергенций
    """
    # Очищаем от NaN
    rsi = rsi.dropna()
    
    if len(rsi) < window * 2:
        return []
    
    divergences = []
    
    # Ищем локальные максимумы и минимумы в RSI
    rsi_peaks = []
    rsi_troughs = []
    
    for i in range(window, len(rsi) - window):
        # Локальный максимум (выше всех соседей в окне)
        if all(rsi.iloc[i] > rsi.iloc[i-j] for j in range(1, window+1)) and \
           all(rsi.iloc[i] > rsi.iloc[i+j] for j in range(1, window+1)):
            rsi_peaks.append(i)
        
        # Локальный минимум (ниже всех соседей в окне)
        if all(rsi.iloc[i] < rsi.iloc[i-j] for j in range(1, window+1)) and \
           all(rsi.iloc[i] < rsi.iloc[i+j] for j in range(1, window+1)):
            rsi_troughs.append(i)
    
    # Проверяем последние 2 пика и впадины, если они есть
    if len(rsi_peaks) >= 2:
        peak1, peak2 = rsi_peaks[-2], rsi_peaks[-1]
        
        # Бычья дивергенция: цена делает более низкий минимум, а RSI - более высокий
        if rsi.iloc[peak2] > rsi.iloc[peak1] and prices.iloc[peak2] < prices.iloc[peak1]:
            divergences.append({
                "type": "bullish",
                "position": peak2,
                "bars_ago": len(rsi) - 1 - peak2
            })
    
    if len(rsi_troughs) >= 2:
        trough1, trough2 = rsi_troughs[-2], rsi_troughs[-1]
        
        # Медвежья дивергенция: цена делает более высокий максимум, а RSI - более низкий
        if rsi.iloc[trough2] < rsi.iloc[trough1] and prices.iloc[trough2] > prices.iloc[trough1]:
            divergences.append({
                "type": "bearish",
                "position": trough2,
                "bars_ago": len(rsi) - 1 - trough2
            })
    
    return divergences


def plot_rsi(prices: pd.Series, rsi: pd.Series, title: str = "RSI Analysis", 
            save_path: Optional[str] = None) -> None:
    """
    Создает график RSI и цены.
    
    Аргументы:
        prices: Series с ценами
        rsi: Series со значениями RSI
        title: Заголовок графика
        save_path: Путь для сохранения графика (если None, график отображается)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # График цены
    ax1.plot(prices.index, prices.values)
    ax1.set_title(title)
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    
    # График RSI
    ax2.plot(rsi.index, rsi.values, color='purple')
    ax2.axhline(y=OVERBOUGHT_LEVEL, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=OVERSOLD_LEVEL, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='k', linestyle='-', alpha=0.2)
    ax2.fill_between(rsi.index, OVERBOUGHT_LEVEL, 100, color='red', alpha=0.1)
    ax2.fill_between(rsi.index, 0, OVERSOLD_LEVEL, color='green', alpha=0.1)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_rsi_summary(dataFrame: pd.DataFrame, period: int = DEFAULT_PERIOD) -> Dict[str, Union[str, float, Dict]]:
    """
    Создает полную сводку по RSI с оценкой и рекомендациями.
    
    Аргументы:
        dataFrame: DataFrame с ценовыми данными
        period: Период для расчета RSI
        
    Возвращает:
        Словарь с полной аналитикой RSI
    """
    try:
        # Расчет стандартного RSI
        rsi_values = current_rsi_call(dataFrame, period)
        
        # Расчет адаптивного RSI
        adaptive_rsi_values = adaptive_rsi(dataFrame['close'], period)
        
        # Получаем сигналы
        standard_signals = get_rsi_signals(rsi_values)
        adaptive_signals = get_rsi_signals(adaptive_rsi_values)
        
        # Текущие значения
        current_rsi = rsi_values.iloc[-1]
        current_adaptive_rsi = adaptive_rsi_values.iloc[-1]
        
        # Формируем рекомендацию на основе обоих индикаторов
        recommendation = "нейтральный"
        
        # Если оба индикатора дают одинаковый сигнал, усиливаем рекомендацию
        if standard_signals["signal"] == adaptive_signals["signal"]:
            if standard_signals["signal"] == "buy":
                recommendation = "сильная покупка" if standard_signals["strength"] >= 2 else "покупка"
            elif standard_signals["signal"] == "sell":
                recommendation = "сильная продажа" if standard_signals["strength"] >= 2 else "продажа"
        
        # Если индикаторы противоречат друг другу, используем более консервативный
        elif standard_signals["signal"] != "neutral" and adaptive_signals["signal"] != "neutral":
            recommendation = "противоречивые сигналы, нужен дополнительный анализ"
        
        # Если только один индикатор дает сигнал
        elif standard_signals["signal"] != "neutral":
            recommendation = standard_signals["signal"]
        elif adaptive_signals["signal"] != "neutral":
            recommendation = adaptive_signals["signal"]
        
        # Рассчитываем силу сигнала (от 1 до 10)
        signal_strength = 5  # Нейтральный старт
        
        # Учитываем состояние перекупленности/перепроданности
        if current_rsi > 80 or current_rsi < 20:
            signal_strength += 2
        elif current_rsi > 70 or current_rsi < 30:
            signal_strength += 1
        
        # Учитываем согласованность стандартного и адаптивного RSI
        if standard_signals["signal"] == adaptive_signals["signal"] and standard_signals["signal"] != "neutral":
            signal_strength += 2
        
        # Учитываем наличие дивергенций
        if standard_signals.get("divergences"):
            for div in standard_signals["divergences"]:
                if div["type"] == "bullish" and "buy" in recommendation:
                    signal_strength += 1
                elif div["type"] == "bearish" and "прода" in recommendation:
                    signal_strength += 1
        
        # Ограничиваем силу сигнала диапазоном от 1 до 10
        signal_strength = max(1, min(10, signal_strength))
        
        # Формируем итоговую сводку
        summary = {
            "standard_rsi": current_rsi,
            "adaptive_rsi": current_adaptive_rsi,
            "state": standard_signals["state"],
            "trend": standard_signals["trend"],
            "recommendation": recommendation,
            "signal_strength": signal_strength,
            "standard_signals": standard_signals,
            "adaptive_signals": adaptive_signals
        }
        
        return summary
        
    except Exception as e:
        raise APK.ApplicationError(f"Ошибка при создании сводки RSI: {str(e)}")


if __name__ == "__main__":
    try:
        # Чтение данных из файла
        data_file = os.path.join(DATA_DIR, "MOEX_2024-11-12_1D_[191120].json")
        
        if os.path.exists(data_file):
            data = pd.read_json(data_file)
        else:
            # Ищем любой JSON файл
            json_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            if json_files:
                data_file = os.path.join(DATA_DIR, json_files[0])
                print(f"Используем файл: {data_file}")
                data = pd.read_json(data_file)
            else:
                # Если файлов нет, создаем тестовые данные
                print("Файлы данных не найдены. Создаем тестовые данные...")
                dates = pd.date_range('2023-01-01', periods=100)
                close = [100 + i + np.random.normal(0, 5) for i in range(100)]
                data = pd.DataFrame({'close': close}, index=dates)
                
                # Создаем колебания для демонстрации RSI
                for i in range(40, 50):
                    data['close'].iloc[i] = data['close'].iloc[i-1] * 1.03  # Рост на 3%
                for i in range(65, 75):
                    data['close'].iloc[i] = data['close'].iloc[i-1] * 0.97  # Падение на 3%
        
        # Проверка наличия необходимого столбца 'close'
        if 'close' not in data.columns:
            raise APK.InvalidInputError("Данные не содержат колонку 'close'")
        
        # Расчет RSI
        standard_rsi = calculate_rsi(data['close'])
        adaptive_rsi = adaptive_rsi(data['close'])
        
        # Получаем сводку
        summary = get_rsi_summary(data)
        
        # Вывод сводки
        print(f"\nСтандартный RSI (период 14): {summary['standard_rsi']:.2f}")
        print(f"Адаптивный RSI: {summary['adaptive_rsi']:.2f}")
        print(f"Состояние: {summary['state']}")
        print(f"Тренд RSI: {summary['trend']}")
        print(f"Рекомендация: {summary['recommendation']}")
        print(f"Сила сигнала: {summary['signal_strength']}/10")
        
        # Отображаем графики
        plots_dir = os.path.join(DATA_DIR, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        plot_rsi(data['close'], standard_rsi, "Стандартный RSI", 
                os.path.join(plots_dir, "standard_rsi.png"))
        plot_rsi(data['close'], adaptive_rsi, "Адаптивный RSI", 
                os.path.join(plots_dir, "adaptive_rsi.png"))
        
        print(f"\nГрафики сохранены в директории: {plots_dir}")
        
    except APK.ApplicationError as e:
        print(f"Ошибка приложения: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")