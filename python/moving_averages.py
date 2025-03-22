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

# Константы для скользящих средних
DEFAULT_SHORT_PERIOD = 9     # Период для короткой SMA/EMA
DEFAULT_LONG_PERIOD = 21     # Период для длинной SMA/EMA
DEFAULT_SIGNAL_PERIOD = 9    # Период для сигнальной линии MACD
TREND_THRESHOLD = 0.01       # Порог для определения силы тренда (%)


def calculate_ma(data: pd.Series, period: int, ma_type: str = 'sma') -> pd.Series:
    """
    Рассчитывает различные типы скользящих средних.
    
    Аргументы:
        data: Серия данных для расчета
        period: Период для скользящего среднего
        ma_type: Тип скользящего среднего ('sma', 'ema', 'wma', 'hull')
        
    Возвращает:
        Series с рассчитанным скользящим средним
    """
    if period <= 0:
        raise APK.InvalidInputError(f"Период должен быть положительным числом, получено: {period}")
        
    if len(data) < period:
        raise APK.InvalidInputError(f"Недостаточно данных для расчета MA. Требуется: {period}, имеется: {len(data)}")
        
    if ma_type.lower() == 'sma':
        # Простое скользящее среднее
        return data.rolling(window=period).mean()
        
    elif ma_type.lower() == 'ema':
        # Экспоненциальное скользящее среднее
        return data.ewm(span=period, adjust=False).mean()
        
    elif ma_type.lower() == 'wma':
        # Взвешенное скользящее среднее (больший вес новым данным)
        weights = np.arange(1, period + 1)
        wma = data.rolling(window=period).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )
        return wma
        
    elif ma_type.lower() == 'hull':
        # Hull MA (более быстрая реакция и меньшая задержка)
        # Hull = WMA[2 * WMA(n/2) - WMA(n)], sqrt(n)
        half_period = period // 2
        
        # Рассчитываем WMA с периодом n/2 и умножаем на 2
        wma_half = calculate_ma(data, half_period, 'wma') * 2
        
        # Вычитаем WMA с периодом n
        wma_full = calculate_ma(data, period, 'wma')
        diff = wma_half - wma_full
        
        # Рассчитываем WMA от разницы с периодом sqrt(n)
        sqrt_period = int(np.sqrt(period))
        hull = calculate_ma(diff, sqrt_period, 'wma')
        
        return hull
        
    else:
        raise APK.InvalidInputError(f"Неизвестный тип скользящего среднего: {ma_type}")


def calculate_macd(data: pd.Series, 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = DEFAULT_SIGNAL_PERIOD) -> pd.DataFrame:
    """
    Рассчитывает индикатор MACD (Moving Average Convergence Divergence).
    
    Аргументы:
        data: Серия данных для расчета
        fast_period: Период быстрой EMA
        slow_period: Период медленной EMA
        signal_period: Период сигнальной линии
        
    Возвращает:
        DataFrame с колонками 'MACD', 'Signal' и 'Histogram'
    """
    # Рассчитываем быструю и медленную EMA
    fast_ema = calculate_ma(data, fast_period, 'ema')
    slow_ema = calculate_ma(data, slow_period, 'ema')
    
    # Рассчитываем линию MACD
    macd_line = fast_ema - slow_ema
    
    # Рассчитываем сигнальную линию (EMA от MACD)
    signal_line = calculate_ma(macd_line, signal_period, 'ema')
    
    # Рассчитываем гистограмму (MACD - Signal)
    histogram = macd_line - signal_line
    
    # Создаем DataFrame с результатами
    macd_df = pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    }, index=data.index)
    
    return macd_df


def current_ma_analysis(dataFrame: pd.DataFrame, 
                      short_period: int = DEFAULT_SHORT_PERIOD, 
                      long_period: int = DEFAULT_LONG_PERIOD) -> pd.DataFrame:
    """
    Добавляет колонки с индикаторами скользящих средних в DataFrame.
    
    Аргументы:
        dataFrame: pandas DataFrame с данными
        short_period: период короткой скользящей средней (по умолчанию 9)
        long_period: период длинной скользящей средней (по умолчанию 21)
    
    Возвращает:
        pandas DataFrame с добавленными колонками:
        - SMA: простая скользящая средняя
        - EMA: экспоненциальная скользящая средняя
        - HMA: Hull MA
        - Volatility: волатильность
        - MA_Signal: сигнал от пересечения (-1, 0, 1)
    """
    # Проверка наличия необходимого столбца
    if 'close' not in dataFrame.columns:
        raise APK.InvalidInputError("Data does not contain 'close' column.")
    
    # Проверка достаточной длины данных
    if len(dataFrame) < long_period:
        raise APK.DatabaseError(f"Not enough data points. Required: {long_period}, Got: {len(dataFrame)}")
    
    # Создаем копию DataFrame
    df = dataFrame.copy()
    
    try:
        # Добавляем различные типы скользящих средних
        df['SMA'] = calculate_ma(df['close'], long_period, 'sma')
        df['EMA'] = calculate_ma(df['close'], short_period, 'ema')
        df['HMA'] = calculate_ma(df['close'], short_period, 'hull')
        
        # Добавляем MACD
        macd_data = calculate_macd(df['close'])
        df['MACD'] = macd_data['MACD']
        df['MACD_Signal'] = macd_data['Signal']
        df['MACD_Histogram'] = macd_data['Histogram']
        
        # Добавляем волатильность (на основе стандартного отклонения)
        df['Volatility'] = df['close'].pct_change().rolling(
            window=short_period).std() * np.sqrt(short_period)
        
        # Добавляем сигналы от пересечения MA
        df['MA_Signal'] = 0  # По умолчанию нет сигнала
        
        # Сигнал на покупку (1): короткая MA пересекает длинную MA снизу вверх
        df.loc[(df['EMA'] > df['SMA']) & 
               (df['EMA'].shift(1) <= df['SMA'].shift(1)), 'MA_Signal'] = 1
        
        # Сигнал на продажу (-1): короткая MA пересекает длинную MA сверху вниз
        df.loc[(df['EMA'] < df['SMA']) & 
               (df['EMA'].shift(1) >= df['SMA'].shift(1)), 'MA_Signal'] = -1
        
        # Усиливаем сигнал, если MACD подтверждает
        # Усиленный сигнал на покупку (2)
        df.loc[(df['MA_Signal'] == 1) & (df['MACD_Histogram'] > 0), 'MA_Signal'] = 2
        
        # Усиленный сигнал на продажу (-2)
        df.loc[(df['MA_Signal'] == -1) & (df['MACD_Histogram'] < 0), 'MA_Signal'] = -2
        
        # Добавляем направление тренда
        df['Trend'] = 0  # Нейтральный по умолчанию
        
        # Определяем восходящий тренд
        df.loc[(df['EMA'] > df['SMA']) & 
               (df['EMA'].shift(5) > df['SMA'].shift(5)) & 
               (df['close'] > df['EMA']), 'Trend'] = 1  # Сильный восходящий
               
        df.loc[(df['EMA'] > df['SMA']) & 
               (df['close'] < df['EMA']), 'Trend'] = 0.5  # Слабый восходящий
        
        # Определяем нисходящий тренд
        df.loc[(df['EMA'] < df['SMA']) & 
               (df['EMA'].shift(5) < df['SMA'].shift(5)) & 
               (df['close'] < df['EMA']), 'Trend'] = -1  # Сильный нисходящий
               
        df.loc[(df['EMA'] < df['SMA']) & 
               (df['close'] > df['EMA']), 'Trend'] = -0.5  # Слабый нисходящий
        
        return df
        
    except Exception as e:
        raise APK.ApplicationError(f"Error calculating MA indicators: {str(e)}")


def get_ma_summary(dataFrame: pd.DataFrame) -> Dict[str, Union[str, float]]:
    """
    Возвращает сводную информацию по индикаторам MA.
    
    Аргументы:
        dataFrame: pandas DataFrame с рассчитанными индикаторами MA
    
    Возвращает:
        dict с краткой информацией о текущем состоянии индикаторов
    """
    try:
        # Проверяем наличие необходимых столбцов
        required_columns = ['close', 'SMA', 'EMA', 'Volatility', 'MA_Signal', 'Trend']
        missing_columns = [col for col in required_columns if col not in dataFrame.columns]
        
        if missing_columns:
            raise APK.InvalidInputError(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
            
        latest = dataFrame.iloc[-1]
        
        # Определяем тренд
        trend_values = {
            1: "Сильный восходящий",
            0.5: "Слабый восходящий",
            0: "Боковой",
            -0.5: "Слабый нисходящий",
            -1: "Сильный нисходящий"
        }
        trend = trend_values.get(latest['Trend'], "Неопределенный")
            
        # Оцениваем волатильность
        avg_volatility = dataFrame['Volatility'].mean()
        current_volatility = latest['Volatility']
        
        if current_volatility > avg_volatility * 1.5:
            volatility_status = "Высокая"
        elif current_volatility < avg_volatility * 0.5:
            volatility_status = "Низкая"
        else:
            volatility_status = "Средняя"
            
        # Оцениваем силу сигнала (от 1 до 10)
        signal_strength = 5  # Нейтральный уровень
        
        # Усиливаем на основе силы тренда
        if latest['Trend'] == 1:
            signal_strength += 2
        elif latest['Trend'] == 0.5:
            signal_strength += 1
        elif latest['Trend'] == -1:
            signal_strength -= 2
        elif latest['Trend'] == -0.5:
            signal_strength -= 1
        
        # Усиливаем на основе согласованности индикаторов
        if latest['EMA'] > latest['SMA'] and latest['MACD'] > latest['MACD_Signal']:
            signal_strength += 1
        elif latest['EMA'] < latest['SMA'] and latest['MACD'] < latest['MACD_Signal']:
            signal_strength -= 1
            
        # Усиливаем на основе пробоя
        if latest['close'] > latest['EMA'] * 1.02 and latest['Trend'] > 0:
            signal_strength += 1
        elif latest['close'] < latest['EMA'] * 0.98 and latest['Trend'] < 0:
            signal_strength -= 1
            
        # Ограничиваем значение от 1 до 10
        signal_strength = max(1, min(10, signal_strength))
        
        # Определяем рекомендацию
        if signal_strength >= 8:
            recommendation = "Сильная покупка"
        elif signal_strength >= 6:
            recommendation = "Покупка"
        elif signal_strength <= 3:
            recommendation = "Сильная продажа"
        elif signal_strength <= 5:
            recommendation = "Продажа"
        else:
            recommendation = "Нейтрально"
            
        # Формируем описание тренда
        price_to_sma_ratio = latest['close'] / latest['SMA'] - 1
        price_to_ema_ratio = latest['close'] / latest['EMA'] - 1
        
        if abs(price_to_sma_ratio) > TREND_THRESHOLD * 2:
            trend_strength = "сильный"
        elif abs(price_to_sma_ratio) > TREND_THRESHOLD:
            trend_strength = "умеренный"
        else:
            trend_strength = "слабый"
            
        direction = "восходящий" if price_to_sma_ratio > 0 else "нисходящий"
        trend_description = f"{trend_strength} {direction}"
        
        if abs(price_to_sma_ratio) < TREND_THRESHOLD / 2:
            trend_description = "боковой"
            
        # Возвращаем результаты
        return {
            'trend': trend,
            'trend_description': trend_description,
            'volatility_status': volatility_status,
            'current_volatility': current_volatility,
            'signal': latest['MA_Signal'],
            'signal_strength': signal_strength,
            'recommendation': recommendation,
            'sma': latest['SMA'],
            'ema': latest['EMA'],
            'close': latest['close'],
            'price_to_sma': price_to_sma_ratio * 100,  # в процентах
            'price_to_ema': price_to_ema_ratio * 100,  # в процентах
            'macd': latest['MACD'] if 'MACD' in latest else None,
            'macd_signal': latest['MACD_Signal'] if 'MACD_Signal' in latest else None,
            'macd_histogram': latest['MACD_Histogram'] if 'MACD_Histogram' in latest else None
        }
        
    except Exception as e:
        raise APK.ApplicationError(f"Error creating MA summary: {str(e)}")


def plot_moving_averages(dataFrame: pd.DataFrame, n_bars: int = 100, 
                       save_path: Optional[str] = None) -> None:
    """
    Создает график со скользящими средними и MACD.
    
    Аргументы:
        dataFrame: DataFrame с рассчитанными MA
        n_bars: Количество последних баров для отображения
        save_path: Путь для сохранения графика (если None, график отображается)
    """
    # Проверяем наличие необходимых данных
    required_columns = ['close', 'SMA', 'EMA']
    missing_columns = [col for col in required_columns if col not in dataFrame.columns]
    if missing_columns:
        print(f"Ошибка: Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
        return
    
    # Получаем последние n_bars
    df = dataFrame.tail(n_bars)
    
    # Создаем фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # График цен и скользящих средних
    ax1.plot(df.index, df['close'], label='Цена', color='blue')
    ax1.plot(df.index, df['SMA'], label='SMA', color='red')
    ax1.plot(df.index, df['EMA'], label='EMA', color='green')
    
    if 'HMA' in df.columns:
        ax1.plot(df.index, df['HMA'], label='HMA', color='purple')
    
    # Отмечаем сигналы на графике
    if 'MA_Signal' in df.columns:
        buy_signals = df[df['MA_Signal'] > 0].index
        sell_signals = df[df['MA_Signal'] < 0].index
        
        ax1.scatter(buy_signals, df.loc[buy_signals, 'close'], marker='^', color='green', s=100, label='Покупка')
        ax1.scatter(sell_signals, df.loc[sell_signals, 'close'], marker='v', color='red', s=100, label='Продажа')
    
    ax1.set_title('Цена и скользящие средние')
    ax1.set_ylabel('Цена')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax2.plot(df.index, df['MACD_Signal'], label='Сигнал', color='red')
        
        # Гистограмма
        if 'MACD_Histogram' in df.columns:
            ax2.bar(df.index, df['MACD_Histogram'], label='Гистограмма', alpha=0.3, color='gray')
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('MACD')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def detect_ma_patterns(dataFrame: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Обнаруживает паттерны скользящих средних.
    
    Аргументы:
        dataFrame: DataFrame с рассчитанными MA
        
    Возвращает:
        Словарь с найденными паттернами и их позициями
    """
    # Проверяем наличие необходимых данных
    required_columns = ['close', 'SMA', 'EMA']
    missing_columns = [col for col in required_columns if col not in dataFrame.columns]
    if missing_columns:
        raise APK.InvalidInputError(f"Отсутствуют необходимые столбцы: {', '.join(missing_columns)}")
    
    # Инициализируем словарь паттернов
    patterns = {
        'ma_crossover_bullish': [],   # Бычье пересечение MA (EMA пересекает SMA снизу вверх)
        'ma_crossover_bearish': [],   # Медвежье пересечение MA (EMA пересекает SMA сверху вниз)
        'price_ma_bounce': [],        # Отскок цены от MA
        'price_ma_breakout': [],      # Пробой MA ценой
        'ma_squeeze': []              # Сужение MA (потенциальный взрыв волатильности)
    }
    
    df = dataFrame.copy()
    
    # Добавляем необходимые вспомогательные столбцы
    df['EMA_SMA_Diff'] = df['EMA'] - df['SMA']
    df['EMA_SMA_Diff_Pct'] = df['EMA_SMA_Diff'] / df['SMA'] * 100
    df['Price_SMA_Diff'] = df['close'] - df['SMA']
    
    # Ищем паттерны
    for i in range(1, len(df)):
        # Бычье пересечение MA
        if df['EMA_SMA_Diff'].iloc[i] > 0 and df['EMA_SMA_Diff'].iloc[i-1] <= 0:
            patterns['ma_crossover_bullish'].append(i)
        
        # Медвежье пересечение MA
        if df['EMA_SMA_Diff'].iloc[i] < 0 and df['EMA_SMA_Diff'].iloc[i-1] >= 0:
            patterns['ma_crossover_bearish'].append(i)
        
        # Отскок цены от MA (цена подходит к MA и отскакивает)
        price_approaching_sma = abs(df['Price_SMA_Diff'].iloc[i-1]) < abs(df['Price_SMA_Diff'].iloc[i-2])
        price_bouncing_from_sma = abs(df['Price_SMA_Diff'].iloc[i]) > abs(df['Price_SMA_Diff'].iloc[i-1])
        
        if price_approaching_sma and price_bouncing_from_sma:
            # Должен быть значительный отскок (минимум 0.5%)
            if abs(df['Price_SMA_Diff'].iloc[i] / df['SMA'].iloc[i]) > 0.005:
                patterns['price_ma_bounce'].append(i)
        
        # Пробой MA ценой (цена пересекает MA при значительном движении)
        price_crosses_sma_up = (df['close'].iloc[i] > df['SMA'].iloc[i] and 
                               df['close'].iloc[i-1] <= df['SMA'].iloc[i-1])
        price_crosses_sma_down = (df['close'].iloc[i] < df['SMA'].iloc[i] and 
                                 df['close'].iloc[i-1] >= df['SMA'].iloc[i-1])
        
        if (price_crosses_sma_up or price_crosses_sma_down) and abs(df['Price_SMA_Diff'].iloc[i] / df['SMA'].iloc[i]) > 0.01:
            patterns['price_ma_breakout'].append(i)
        
        # Сужение MA (EMA и SMA сближаются)
        if i > 5:
            avg_diff_before = abs(df['EMA_SMA_Diff_Pct'].iloc[i-5:i-1]).mean()
            current_diff = abs(df['EMA_SMA_Diff_Pct'].iloc[i])
            
            if current_diff < avg_diff_before * 0.3 and current_diff < 0.2:  # Значительное сужение
                patterns['ma_squeeze'].append(i)
    
    return patterns


if __name__ == "__main__":
    try:
        # Поиск файла данных
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
                    # Добавляем циклы
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
                test_file = os.path.join(DATA_DIR, "test_ma_data.json")
                test_data.to_json(test_file)
                print(f"Тестовые данные сохранены в {test_file}")
        
        # Загрузка данных
        data = pd.read_json(test_file)
        
        # Добавление индикаторов MA
        data_with_ma = current_ma_analysis(data)
        
        # Получение сводки
        summary = get_ma_summary(data_with_ma)
        
        # Обнаружение паттернов
        patterns = detect_ma_patterns(data_with_ma)
        
        # Вывод результатов
        print("\nПоследние 10 значений индикаторов MA:")
        print(data_with_ma[['close', 'SMA', 'EMA', 'Volatility', 'MA_Signal', 'Trend']].tail(10))
        
        print("\nСводная информация:")
        print(f"Тренд: {summary['trend']} ({summary['trend_description']})")
        print(f"Волатильность: {summary['volatility_status']} ({summary['current_volatility']:.2%})")
        print(f"Сигнал: {summary['signal']}")
        print(f"Сила сигнала: {summary['signal_strength']}/10")
        print(f"Рекомендация: {summary['recommendation']}")
        print(f"Цена закрытия: {summary['close']:.2f}")
        print(f"SMA: {summary['sma']:.2f}")
        print(f"EMA: {summary['ema']:.2f}")
        print(f"Отношение цены к SMA: {summary['price_to_sma']:.2f}%")
        print(f"Отношение цены к EMA: {summary['price_to_ema']:.2f}%")
        
        if summary['macd'] is not None:
            print(f"MACD: {summary['macd']:.4f}")
            print(f"MACD Signal: {summary['macd_signal']:.4f}")
            print(f"MACD Histogram: {summary['macd_histogram']:.4f}")
        
        # Вывод паттернов
        print("\nОбнаруженные паттерны (последние 20 баров):")
        recent_patterns = []
        bar_count = min(20, len(data_with_ma))
        recent_range = list(range(len(data_with_ma) - bar_count, len(data_with_ma)))
        
        for pattern_name, indices in patterns.items():
            recent_indices = [i for i in indices if i in recent_range]
            if recent_indices:
                positions = [len(data_with_ma) - i for i in reversed(recent_indices)]
                recent_patterns.append(f"{pattern_name}: {positions} баров назад")
        
        if recent_patterns:
            for pattern in recent_patterns:
                print(f"- {pattern}")
        else:
            print("Паттерны не обнаружены")
        
        # Построение графика
        plots_dir = os.path.join(DATA_DIR, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        plot_moving_averages(data_with_ma, save_path=os.path.join(plots_dir, "moving_averages.png"))
        print(f"\nГрафик сохранен в {plots_dir}/moving_averages.png")
        
    except APK.ApplicationError as e:
        print(f"Ошибка: {str(e)}")
    except Exception as e:
        print(f"Непредвиденная ошибка: {str(e)}")