"""
Главный модуль системы технического анализа.
Объединяет все индикаторы, собирает результаты и генерирует итоговую рекомендацию.
"""

import _AppProjectKit as kit
import os
import pandas as pd
import numpy as np
from colorama import Fore, Back, Style, init
from datetime import datetime, timedelta
import pytz
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any

# Импорт модулей анализа
from rsi_call import current_rsi_call, get_rsi_summary
from bollinger_strategy import bollinger_strings
from candlestick_patterns import current_candlestick_patterns, get_pattern_signals
from supres_levels import today_levels, find_key_levels
from data_collector import ask_moex
from moving_averages import current_ma_analysis, get_ma_summary
from Stochastic_RSI import calculate_stochastic_rsi, get_stoch_rsi_summary
from volume import volume_analysis, get_volume_summary, find_support_resistance

# Директория для хранения данных
DATA_DIR = "storage"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Инициализация colorama
init(autoreset=True)

# Баннер
try:
    with open("banner.txt", 'r', encoding="UTF-8") as banner_file:
        banner = banner_file.read()
except FileNotFoundError:
    banner = """
    ╔═══════════════════════════════════════════════╗
    ║           СИСТЕМА ТЕХНИЧЕСКОГО АНАЛИЗА        ║
    ╚═══════════════════════════════════════════════╝
    """

# Московское время
MOSCOW_TZ = pytz.timezone('Europe/Moscow')

# Веса индикаторов для итоговой рекомендации
INDICATOR_WEIGHTS = {
    "Support & Resistance": 30.0,
    "Volume Analysis": 27.9,
    "RSI": 18.9,
    "Bollinger Bands": 16.1,
    "Stochastic RSI": 15.0,
    "Moving Averages": 11.8,
    "Candlestick Patterns": 5.0
}

def load_config():
    """Загружает конфигурацию из JSON файла."""
    try:
        with open('CONFIG.json', 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        # Создаем конфигурацию по умолчанию
        default_config = {
            "ticker": "SBER",
            "interval": 10,
            "filename": "auto",
            "instant_processing": True,
            "self_creation": True
        }
        
        # Сохраняем конфигурацию
        with open('CONFIG.json', 'w', encoding='utf-8') as file:
            json.dump(default_config, file, indent=4)
            
        print(f"{Fore.YELLOW}Внимание: Файл CONFIG.json не найден. Создана конфигурация по умолчанию.")
        return default_config


def normalize_signal(value: float, min_val: float = -2, max_val: float = 2) -> float:
    """
    Нормализует значение сигнала в диапазон от -1 до 1.
    
    Аргументы:
        value: Значение для нормализации
        min_val: Минимальное значение в исходном диапазоне
        max_val: Максимальное значение в исходном диапазоне
        
    Возвращает:
        Нормализованное значение от -1 до 1
    """
    # Защита от деления на ноль
    if max_val == min_val:
        return 0
    
    # Нормализация
    normalized = (value - min_val) / (max_val - min_val) * 2 - 1
    
    # Ограничение диапазоном [-1, 1]
    return max(-1, min(1, normalized))

def calculate_weighted_recommendation(
    support_res_signal: float,
    volume_signal: float,
    rsi_signal: float,
    bollinger_signal: float,
    stoch_rsi_signal: float,
    ma_signal: float,
    candlestick_signal: float
) -> Dict[str, Any]:
    """
    Рассчитывает взвешенную рекомендацию на основе сигналов индикаторов.
    
    Возвращает:
        Словарь с итоговой рекомендацией и вкладом каждого индикатора
    """
    # Нормализуем сигналы
    normalized_signals = {
        "Support & Resistance": normalize_signal(support_res_signal, -2, 2),
        "Volume Analysis": normalize_signal(volume_signal, -2, 2),
        "RSI": normalize_signal(rsi_signal, -10, 10),
        "Bollinger Bands": normalize_signal(bollinger_signal, -2, 2),
        "Stochastic RSI": normalize_signal(stoch_rsi_signal, -10, 10),
        "Moving Averages": normalize_signal(ma_signal, -2, 2),
        "Candlestick Patterns": normalize_signal(candlestick_signal, -1, 1)
    }
    
    # Вклад каждого индикатора
    contributions = {}
    total_weight = sum(INDICATOR_WEIGHTS.values())
    
    for indicator, signal in normalized_signals.items():
        weight = INDICATOR_WEIGHTS[indicator]
        contribution = (signal * weight) / total_weight
        contributions[indicator] = contribution
    
    # Итоговый взвешенный сигнал
    weighted_signal = sum(contributions.values())
    
    # Определяем силу сигнала (от 0 до 10)
    signal_strength = abs(weighted_signal) * 10
    
    # Формируем рекомендацию
    if weighted_signal > 0.6:
        recommendation = "Сильная покупка"
    elif weighted_signal > 0.3:
        recommendation = "Покупка"
    elif weighted_signal > 0.1:
        recommendation = "Слабая покупка"
    elif weighted_signal < -0.6:
        recommendation = "Сильная продажа"
    elif weighted_signal < -0.3:
        recommendation = "Продажа"
    elif weighted_signal < -0.1:
        recommendation = "Слабая продажа"
    else:
        recommendation = "Нейтрально"
    
    return {
        "recommendation": recommendation,
        "signal": weighted_signal,
        "strength": signal_strength,
        "contributions": contributions,
        "normalized_signals": normalized_signals
    }


def generate_detailed_recommendation(
    weighted_signal: float,
    signal_strength: float,
    contributions: Dict[str, float],
    current_price: float,
    levels: kit.todaySupRes,
    rsi_value: float,
    volatility: float,
    trend_description: str,
    stoch_rsi_condition: str
) -> Dict[str, Any]:
    """
    Генерирует подробную рекомендацию на основе всех индикаторов и их вкладов.
    
    Аргументы:
        weighted_signal: Взвешенный сигнал от -1 до 1
        signal_strength: Сила сигнала от 0 до 10
        contributions: Словарь с вкладами каждого индикатора
        current_price: Текущая цена
        levels: Объект с уровнями поддержки и сопротивления
        rsi_value: Текущее значение RSI
        volatility: Текущая волатильность (в процентах)
        trend_description: Описание тренда
        stoch_rsi_condition: Состояние стохастического RSI
        
    Возвращает:
        Словарь с детальной рекомендацией
    """
    # Базовая рекомендация на основе взвешенного сигнала
    if weighted_signal > 0.6:
        base_recommendation = "Сильная покупка"
        action = "покупать"
    elif weighted_signal > 0.3:
        base_recommendation = "Покупка"
        action = "покупать"
    elif weighted_signal > 0.1:
        base_recommendation = "Слабая покупка"
        action = "рассмотреть возможность покупки"
    elif weighted_signal < -0.6:
        base_recommendation = "Сильная продажа"
        action = "продавать"
    elif weighted_signal < -0.3:
        base_recommendation = "Продажа"
        action = "продавать"
    elif weighted_signal < -0.1:
        base_recommendation = "Слабая продажа"
        action = "рассмотреть возможность продажи"
    else:
        base_recommendation = "Нейтрально"
        action = "удерживать текущие позиции"
    
    # Находим ближайшие уровни поддержки и сопротивления
    closest_support = max(
        [levels.support_1, levels.support_2, levels.support_3], 
        key=lambda x: x if x < current_price else 0
    )
    closest_resistance = min(
        [levels.resistance_1, levels.resistance_2, levels.resistance_3], 
        key=lambda x: x if x > current_price else float('inf')
    )
    
    # Рассчитываем рекомендуемые уровни стоп-лосса и тейк-профита
    if action in ["покупать", "рассмотреть возможность покупки"]:
        stop_loss = closest_support * 0.995  # 0.5% ниже поддержки
        take_profit_conservative = closest_resistance
        take_profit_aggressive = current_price * (1 + (current_price - closest_support) / current_price * 2)
    elif action in ["продавать", "рассмотреть возможность продажи"]:
        stop_loss = closest_resistance * 1.005  # 0.5% выше сопротивления
        take_profit_conservative = closest_support
        take_profit_aggressive = current_price * (1 - (closest_resistance - current_price) / current_price * 2)
    else:
        stop_loss = None
        take_profit_conservative = None
        take_profit_aggressive = None
    
    # Оцениваем временной горизонт для сделки
    if volatility > 0.05:  # Высокая волатильность
        timeframe = "краткосрочный (1-3 дня)"
    elif "сильный" in trend_description.lower():
        timeframe = "среднесрочный (1-3 недели)"
    else:
        timeframe = "среднесрочный (1-2 недели)"
    
    # Оцениваем риски
    risks = []
    
    if rsi_value > 70:
        risks.append("RSI указывает на перекупленность рынка")
    elif rsi_value < 30:
        risks.append("RSI указывает на перепроданность рынка")
        
    if stoch_rsi_condition == "Перекуплен":
        risks.append("Стохастический RSI в зоне перекупленности")
    elif stoch_rsi_condition == "Перепродан":
        risks.append("Стохастический RSI в зоне перепроданности")
    
    if volatility > 0.05:
        risks.append(f"Повышенная волатильность ({volatility:.2%})")
    
    # Оцениваем благоприятные факторы
    positives = []
    negative_contribs = {k: v for k, v in contributions.items() if v < 0}
    positive_contribs = {k: v for k, v in contributions.items() if v > 0}
    
    # Сортируем по абсолютному значению вклада (от большего к меньшему)
    top_positives = sorted(positive_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_negatives = sorted(negative_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Добавляем 3 наиболее значимых положительных фактора
    for indicator, value in top_positives[:3]:
        if indicator == "Support & Resistance":
            positives.append(f"Благоприятные уровни поддержки и сопротивления")
        elif indicator == "Volume Analysis":
            positives.append(f"Объемы подтверждают движение цены")
        elif indicator == "RSI":
            if rsi_value > 50 and action in ["покупать", "рассмотреть возможность покупки"]:
                positives.append(f"RSI показывает силу бычьего тренда ({rsi_value:.1f})")
            elif rsi_value < 50 and action in ["продавать", "рассмотреть возможность продажи"]:
                positives.append(f"RSI показывает силу медвежьего тренда ({rsi_value:.1f})")
        elif indicator == "Bollinger Bands":
            positives.append(f"Благоприятный сигнал от полос Боллинджера")
        elif indicator == "Stochastic RSI":
            positives.append(f"Благоприятный сигнал от стохастического RSI")
        elif indicator == "Moving Averages":
            positives.append(f"Подтверждение от скользящих средних")
        elif indicator == "Candlestick Patterns":
            positives.append(f"Благоприятные свечные паттерны")
    
    # Формируем подробную рекомендацию
    detailed_recommendation = {
        "base_recommendation": base_recommendation,
        "signal_strength": signal_strength,
        "action": action,
        "timeframe": timeframe,
        "stop_loss": stop_loss,
        "take_profit_conservative": take_profit_conservative,
        "take_profit_aggressive": take_profit_aggressive,
        "risks": risks,
        "positives": positives,
        "rationale": f"Рекомендация основана на комплексном анализе {len(contributions)} индикаторов, с наибольшим положительным вкладом от {top_positives[0][0] if top_positives else 'нет положительных факторов'} и наибольшим отрицательным вкладом от {top_negatives[0][0] if top_negatives else 'нет отрицательных факторов'}."
    }
    
    # Формируем текстовую рекомендацию
    text_recommendation = f"""
РЕКОМЕНДАЦИЯ: {base_recommendation.upper()} (сила сигнала: {signal_strength:.1f}/10)

Действие: {action.capitalize()}
Временной горизонт: {timeframe}
"""
    
    if stop_loss is not None:
        text_recommendation += f"""
Рекомендуемые уровни:
- Стоп-лосс: {stop_loss:.2f}
- Тейк-профит (консервативный): {take_profit_conservative:.2f}
- Тейк-профит (агрессивный): {take_profit_aggressive:.2f}
"""
    
    if positives:
        text_recommendation += "\nБлагоприятные факторы:\n"
        for pos in positives:
            text_recommendation += f"✓ {pos}\n"
    
    if risks:
        text_recommendation += "\nРиски и предупреждения:\n"
        for risk in risks:
            text_recommendation += f"! {risk}\n"
    
    text_recommendation += f"\nОбоснование: {detailed_recommendation['rationale']}"
    
    detailed_recommendation["text"] = text_recommendation
    
    return detailed_recommendation

def process_data_frame(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Обрабатывает DataFrame и генерирует аналитику.
    
    Аргументы:
        data: DataFrame с данными
        
    Возвращает:
        Кортеж (обработанный DataFrame, словарь с рекомендациями)
    """
    print(banner)

    # Создание проанализированного фрейма
    data_mod = data.copy(deep=True)

    # Уровни поддержки и сопротивления
    levels = today_levels(data)
    key_levels = find_key_levels(data)
    
    # Преобразуем сигнал уровней в числовое значение
    # Если цена близка к поддержке, это сигнал на покупку (+1, +2)
    # Если цена близка к сопротивлению, это сигнал на продажу (-1, -2)
    current_price = data['close'].iloc[-1]
    closest_support = min(
        [levels.support_1, levels.support_2, levels.support_3], 
        key=lambda x: abs(current_price - x)
    )
    closest_resistance = min(
        [levels.resistance_1, levels.resistance_2, levels.resistance_3], 
        key=lambda x: abs(current_price - x)
    )
    
    support_distance = (current_price - closest_support) / current_price
    resistance_distance = (closest_resistance - current_price) / current_price
    
    if support_distance < resistance_distance:
        # Ближе к поддержке
        if support_distance < 0.01:  # В пределах 1%
            support_res_signal = 2
        elif support_distance < 0.03:  # В пределах 3%
            support_res_signal = 1
        else:
            support_res_signal = 0
    else:
        # Ближе к сопротивлению
        if resistance_distance < 0.01:  # В пределах 1%
            support_res_signal = -2
        elif resistance_distance < 0.03:  # В пределах 3%
            support_res_signal = -1
        else:
            support_res_signal = 0

    # Анализ объемов
    volume_result = volume_analysis(data_mod, levels.support_1, levels.resistance_1)
    data_mod = volume_result  # Обновляем DataFrame с результатами анализа объемов
    volume_summary = get_volume_summary(data_mod)
    volume_signal = volume_summary['signal']

    # Добавляем RSI
    data_mod["RSI"] = current_rsi_call(data, 14)
    rsi_summary = get_rsi_summary(data_mod)
    rsi_signal = rsi_summary['signal_strength']
    if "recommendation" in rsi_summary and "прода" in rsi_summary['recommendation'].lower():
        rsi_signal = -rsi_signal  # Инверсия для продажи

    # Боллинджер
    bollinger = bollinger_strings(data)
    if bollinger:
        data_mod["[B]top"] = bollinger.dataFrame["Upper"]
        data_mod["[B]bottom"] = bollinger.dataFrame["Lower"]
        data_mod["[B]cue"] = bollinger.dataFrame["Signal"]
        
        # Определяем сигнал из рекомендации
        if "покупать" in bollinger.recommendation.lower():
            bollinger_signal = 2 if "сильная" in bollinger.recommendation.lower() else 1
        elif "продавать" in bollinger.recommendation.lower():
            bollinger_signal = -2 if "сильная" in bollinger.recommendation.lower() else -1
        else:
            bollinger_signal = 0
    else:
        bollinger_signal = 0

    # Свечные паттерны
    candlestick = current_candlestick_patterns(data)
    data_mod["[C]Hammer"] = candlestick["Hammer"]
    data_mod['[C]HangingMan'] = candlestick['HangingMan']
    data_mod['[C]Engulfing'] = candlestick['Engulfing']
    
    pattern_signals = get_pattern_signals(candlestick)
    if "бычий" in pattern_signals['overall'].lower():
        candlestick_signal = 1 if "сильный" in pattern_signals['overall'].lower() else 0.5
    elif "медвежий" in pattern_signals['overall'].lower():
        candlestick_signal = -1 if "сильный" in pattern_signals['overall'].lower() else -0.5
    else:
        candlestick_signal = 0

    # Скользящие средние
    data_mod = current_ma_analysis(data_mod)
    ma_summary = get_ma_summary(data_mod)
    ma_signal = ma_summary['signal']

    # Стохастический RSI
    data_mod = calculate_stochastic_rsi(data_mod)
    stoch_rsi_summary = get_stoch_rsi_summary(data_mod)
    
    # Определяем сигнал из рекомендации
    if "покупка" in stoch_rsi_summary['recommendation'].lower():
        stoch_rsi_signal = stoch_rsi_summary['signal_strength']
    elif "продажа" in stoch_rsi_summary['recommendation'].lower():
        stoch_rsi_signal = -stoch_rsi_summary['signal_strength']
    else:
        stoch_rsi_signal = 0

    # Расчет итоговой взвешенной рекомендации
    weighted_recommendation = calculate_weighted_recommendation(
        support_res_signal,
        volume_signal,
        rsi_signal,
        bollinger_signal,
        stoch_rsi_signal,
        ma_signal,
        candlestick_signal
    )

    # Дополняем детальной рекомендацией
    detailed_recommendation = generate_detailed_recommendation(
        weighted_recommendation["signal"],
        weighted_recommendation["strength"],
        weighted_recommendation["contributions"],
        data['close'].iloc[-1],
        levels,
        data_mod["RSI"].iloc[-1],
        ma_summary['current_volatility'],
        ma_summary['trend_description'] if 'trend_description' in ma_summary else ma_summary['trend'],
        stoch_rsi_summary['condition']
    )

    # Обновляем weighted_recommendation с детальной информацией
    weighted_recommendation.update(detailed_recommendation)

    # Вывод результатов
    print(Fore.CYAN + "Уровни поддержки и сопротивления:")
    print(Fore.CYAN + str(levels) + "\n")

    if bollinger:
        print(Fore.CYAN + "Рекомендация Боллинджера:")
        print(Fore.CYAN + bollinger.recommendation + "\n")

    print(Fore.CYAN + "Сводная информация по Скользящим средним:")
    print(Fore.CYAN + f"Тренд: {ma_summary['trend']}")
    print(Fore.CYAN + f"Волатильность: {ma_summary['volatility_status']} ({ma_summary['current_volatility']:.2%})")
    print(Fore.CYAN + f"Сигнал: {ma_summary['signal']}")
    print(Fore.CYAN + f"Цена закрытия: {ma_summary['close']:.2f}")
    print(Fore.CYAN + f"SMA: {ma_summary['sma']:.2f}")
    print(Fore.CYAN + f"EMA: {ma_summary['ema']:.2f}\n")

    print(Fore.CYAN + "Текущее состояние по Стохастик-RSI:")
    print(Fore.CYAN + f"Состояние: {stoch_rsi_summary['condition']}")
    print(Fore.CYAN + f"Тренд: {stoch_rsi_summary['trend']}")
    print(Fore.CYAN + f"RSI: {stoch_rsi_summary['rsi']:.2f}")
    print(Fore.CYAN + f"Стохастик %K: {stoch_rsi_summary['stoch_k']:.2f}")
    print(Fore.CYAN + f"Стохастик %D: {stoch_rsi_summary['stoch_d']:.2f}\n")

    print(Fore.CYAN + "Текущее состояние по Объёмам:")
    print(Fore.CYAN + f"Тренд объема: {volume_summary['volume_trend']}")
    print(Fore.CYAN + f"Текущий объем: {volume_summary['current_volume']:,.0f}")
    print(Fore.CYAN + f"Изменение объема: {volume_summary['volume_change']:.2f}%")
    print(Fore.CYAN + f"Сигнал ({volume_summary['signal']}): {volume_summary['signal_description']}\n")

    # Вывод итоговой рекомендации
    print(Fore.GREEN + "=========== ИТОГОВАЯ РЕКОМЕНДАЦИЯ ===========")
    print(Fore.GREEN + detailed_recommendation["text"])
    print(Fore.GREEN + "==========================================\n")

    # Вывод вклада каждого индикатора
    print(Fore.YELLOW + "Вклад индикаторов:")
    for indicator, contribution in weighted_recommendation['contributions'].items():
        color = Fore.GREEN if contribution > 0 else Fore.RED if contribution < 0 else Fore.WHITE
        print(color + f"{indicator}: {contribution:.4f}")

    print(Fore.BLACK + Back.WHITE + 'Анализ свечей тикера выполнен в {}'.format(datetime.now(MOSCOW_TZ)))

    # Возвращаем обработанный DataFrame и результаты анализа
    return data_mod.drop(columns=["prev_close", "prev_volume"] if "prev_close" in data_mod.columns else []), weighted_recommendation

def plot_summary(data: pd.DataFrame, recommendation: Dict[str, Any], 
                save_path: Optional[str] = None) -> None:
    """
    Создает сводный график с основными индикаторами и рекомендацией.
    
    Аргументы:
        data: DataFrame с данными и рассчитанными индикаторами
        recommendation: Словарь с рекомендацией
        save_path: Путь для сохранения графика
    """
    # Создаем фигуру с подграфиками
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Последние 100 баров или меньше
    n_bars = min(100, len(data))
    df = data.tail(n_bars)
    
    # График цены с MA и Bollinger Bands
    axs[0].plot(df.index, df['close'], label='Цена', color='blue')
    
    if 'SMA' in df.columns:
        axs[0].plot(df.index, df['SMA'], label='SMA', color='red', alpha=0.7)
    
    if 'EMA' in df.columns:
        axs[0].plot(df.index, df['EMA'], label='EMA', color='green', alpha=0.7)
    
    if '[B]top' in df.columns and '[B]bottom' in df.columns:
        axs[0].plot(df.index, df['[B]top'], label='Верхняя полоса Боллинджера', 
                   color='purple', linestyle='--', alpha=0.5)
        axs[0].plot(df.index, df['[B]bottom'], label='Нижняя полоса Боллинджера', 
                   color='purple', linestyle='--', alpha=0.5)
        axs[0].fill_between(df.index, df['[B]top'], df['[B]bottom'], color='purple', alpha=0.05)
    
    axs[0].set_title(f'Цена и технические индикаторы | Рекомендация: {recommendation["base_recommendation"]}')
    axs[0].set_ylabel('Цена')
    axs[0].legend(loc='upper left')
    axs[0].grid(True, alpha=0.3)
    
    # График RSI
    if 'RSI' in df.columns:
        axs[1].plot(df.index, df['RSI'], label='RSI', color='darkorange')
        axs[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
        axs[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
        axs[1].set_ylabel('RSI')
        axs[1].set_ylim(0, 100)
        axs[1].grid(True, alpha=0.3)
    
    # График Стохастического RSI
    if 'StochRSI_K' in df.columns and 'StochRSI_D' in df.columns:
        axs[2].plot(df.index, df['StochRSI_K'], label='%K', color='blue')
        axs[2].plot(df.index, df['StochRSI_D'], label='%D', color='red')
        axs[2].axhline(y=80, color='red', linestyle='--', alpha=0.5)
        axs[2].axhline(y=20, color='green', linestyle='--', alpha=0.5)
        axs[2].set_ylabel('Stochastic RSI')
        axs[2].set_ylim(0, 100)
        axs[2].legend(loc='upper left')
        axs[2].grid(True, alpha=0.3)
    
    # График объемов
    if 'volume' in df.columns:
        axs[3].bar(df.index, df['volume'], label='Объем', color='blue', alpha=0.6)
        if 'volume_ma' in df.columns:
            axs[3].plot(df.index, df['volume_ma'], label='MA объема', color='red')
        axs[3].set_ylabel('Объем')
        axs[3].legend(loc='upper left')
        axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Добавляем аннотацию с ключевыми рекомендациями
    if 'action' in recommendation and 'timeframe' in recommendation:
        action_text = f"Действие: {recommendation['action'].capitalize()}"
        timeframe_text = f"Горизонт: {recommendation['timeframe']}"
        stop_loss_text = ""
        take_profit_text = ""
        
        if 'stop_loss' in recommendation and recommendation['stop_loss'] is not None:
            stop_loss_text = f"Стоп-лосс: {recommendation['stop_loss']:.2f}"
            
        if 'take_profit_conservative' in recommendation and recommendation['take_profit_conservative'] is not None:
            take_profit_text = f"Тейк-профит: {recommendation['take_profit_conservative']:.2f}"
        
        annotation_text = f"{action_text}\n{timeframe_text}"
        if stop_loss_text:
            annotation_text += f"\n{stop_loss_text}"
        if take_profit_text:
            annotation_text += f"\n{take_profit_text}"
        
        # Добавляем аннотацию в верхний правый угол
        axs[0].annotate(annotation_text, 
                       xy=(0.98, 0.98),
                       xycoords='axes fraction',
                       ha='right', va='top',
                       bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3))
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    """Основная функция программы"""
    # Добавление в видимость
    global TICKER
    global INTERVAL
    global FILENAME
    global INSTANT_PROCESSING
    global SELF_CREATION
    global DATA_DIR
    global MOSCOW_TZ

    # Загрузка конфигурации
    CONFIG = load_config()
    TICKER = CONFIG.get("ticker", "SBER")
    INTERVAL = CONFIG.get("interval", 10)
    FILENAME = CONFIG.get("filename", "auto")
    INSTANT_PROCESSING = CONFIG.get("instant_processing", True)
    SELF_CREATION = CONFIG.get("self_creation", True)

    # Определение имени файла
    if FILENAME == "auto":
        now = datetime.now(MOSCOW_TZ)
        FILENAME = f"{TICKER}_{now.strftime('%Y-%m-%d')}_{INTERVAL}m_[{now.strftime('%H%M%S')}].json"

    combined_path = os.path.join(DATA_DIR, FILENAME)

    # Загрузка или создание данных
    if SELF_CREATION:
        print(f"{Fore.YELLOW}Загрузка данных с MOEX для {TICKER}...")
        data = ask_moex(ticker=TICKER, interval=INTERVAL, period="1D", record=True)
        print(f"{Fore.GREEN}Данные MOEX загружены успешно!")
    else:
        if os.path.exists(combined_path):
            data = pd.read_json(combined_path)
        else:
            print(f"{Fore.RED}Файл {combined_path} не найден. Загрузка данных с MOEX...")
            data = ask_moex(ticker=TICKER, interval=INTERVAL, period="1D", record=True)

    print(Fore.BLACK + Back.WHITE + 'Последние свечи в базе\n', data.tail(4), "\n")

    if INSTANT_PROCESSING:
        # Обработка данных
        processed_data, recommendation = process_data_frame(data)
        output_filename = os.path.splitext(FILENAME)[0] + "_processed.json"
        output_path = os.path.join(DATA_DIR, output_filename)
        processed_data.to_json(output_path)
        
        # Создание графика
        plots_dir = os.path.join(DATA_DIR, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        plot_path = os.path.join(plots_dir, f"{TICKER}_summary_{datetime.now(MOSCOW_TZ).strftime('%Y%m%d_%H%M%S')}.png")
        plot_summary(processed_data, recommendation, plot_path)
        print(f"{Fore.GREEN}График сохранен: {plot_path}")
        
        print(f"{Fore.GREEN}Обработанные данные сохранены: {output_path}")
    else:
        # Режим постоянного мониторинга
        # Временные границы торгового дня
        market_open = datetime.now(MOSCOW_TZ).replace(hour=10, minute=0, second=0, microsecond=0)
        market_close = datetime.now(MOSCOW_TZ).replace(hour=18, minute=50, second=0, microsecond=0)
        
        print(f"{Fore.YELLOW}Запуск мониторинга с {market_open.strftime('%H:%M')} до {market_close.strftime('%H:%M')}")
        
        while datetime.now(MOSCOW_TZ) < market_close:
            now = datetime.now(MOSCOW_TZ)
            
            if now >= market_open and now.minute % INTERVAL == 0 and now.second == 0:
                try:
                    print(f"{Fore.CYAN}Запрос данных с MOEX в {now.strftime('%H:%M:%S')}...")
                    upcomingCandle = ask_moex(ticker=TICKER, interval=INTERVAL, period="1D", record=False).iloc[-1]
                    
                    if upcomingCandle.empty:
                        print(f"{Fore.RED}Получена пустая свеча, пропускаем обновление.")
                        continue
                    
                    # Добавляем новую свечу в данные
                    data.loc[len(data)] = upcomingCandle
                    
                    # Обрабатываем данные
                    processed_data, recommendation = process_data_frame(data)
                    
                    # Сохраняем обработанные данные
                    output_filename = os.path.splitext(FILENAME)[0] + "_processed.json"
                    output_path = os.path.join(DATA_DIR, output_filename)
                    processed_data.to_json(output_path)
                    
                    # Создаем график
                    plots_dir = os.path.join(DATA_DIR, "plots")
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    
                    plot_path = os.path.join(plots_dir, f"{TICKER}_summary_{now.strftime('%Y%m%d_%H%M%S')}.png")
                    plot_summary(processed_data, recommendation, plot_path)
                    
                    print(f"{Fore.GREEN}Данные обновлены в {now.strftime('%H:%M:%S')}")
                    print(f"{Fore.GREEN}График сохранен: {plot_path}")
                    
                except Exception as e:
                    print(f"{Fore.RED}Ошибка при обновлении данных: {e}")
                
                time.sleep(10)  # Короткая пауза

if __name__ == "__main__":
    try:
        main()
    except kit.ApplicationError as e:
        print(f"{Fore.RED}Ошибка приложения: {e}")
    except Exception as e:
        print(f"{Fore.RED}Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()