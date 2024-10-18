import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bollinger_strategy import (
    calculate_bollinger_bands,
    generate_signals,
    calculate_returns,
    plot_bollinger_bands,
    plot_strategy_performance,
    generate_recommendation
)

class TestBollingerStrategy(unittest.TestCase):

    def setUp(self):
        # Генерируем цены акций с использованием геометрического броуновского движения (GBM)
        # Параметры GBM
        S0 = np.random.uniform(50, 150)        # Начальная цена акции (от $50 до $150)
        mu = np.random.uniform(-0.001, 0.001)  # Коэффициент дрейфа (от -0.1% до 0.1%)
        sigma = np.random.uniform(0.005, 0.02) # Волатильность (от 0.5% до 2%)
        T = 100                                # Количество временных шагов (например, дней)
        dt = 1                                 # Временной интервал

        np.random.seed()

        # Генерируем ежедневные доходности
        returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=T))

        # Генерируем ценовой ряд
        price_series = S0 * returns.cumprod()

        # Генерируем даты
        dates = pd.date_range(start='2023-01-01', periods=T, freq='D')

        # Создаем DataFrame с добавлением случайных отклонений для Open, High, Low
        self.data = pd.DataFrame({
            'Date': dates,
            'Open': price_series * np.random.uniform(0.99, 1.01, size=T),
            'High': price_series * np.random.uniform(1.00, 1.02, size=T),
            'Low': price_series * np.random.uniform(0.98, 1.00, size=T),
            'Close': price_series,
            'Volume': np.random.randint(1000, 10000, size=T)
        })
        self.data.set_index('Date', inplace=True)

    def test_calculate_bollinger_bands(self):
        """Тестируем функцию calculate_bollinger_bands."""
        data_with_bands = calculate_bollinger_bands(self.data.copy(), window=20, k=2)
        # Проверяем наличие столбцов
        self.assertIn('SMA', data_with_bands.columns)
        self.assertIn('Upper', data_with_bands.columns)
        self.assertIn('Lower', data_with_bands.columns)
        # Проверяем, что значения не содержат NaN после периода window
        self.assertFalse(data_with_bands['SMA'].isnull().iloc[20:].any())

    def test_generate_signals(self):
        """Тестируем функцию generate_signals."""
        data_with_bands = calculate_bollinger_bands(self.data.copy())
        data_with_signals = generate_signals(data_with_bands)
        # Проверяем наличие столбцов
        self.assertIn('Signal', data_with_signals.columns)
        self.assertIn('Position', data_with_signals.columns)
        # Проверяем значения сигналов
        self.assertTrue(data_with_signals['Signal'].isin([-1, 0, 1]).all())

    def test_calculate_returns(self):
        """Тестируем функцию calculate_returns."""
        data_with_bands = calculate_bollinger_bands(self.data.copy())
        data_with_signals = generate_signals(data_with_bands)
        data_with_returns = calculate_returns(data_with_signals)
        # Проверяем наличие столбцов
        self.assertIn('Returns', data_with_returns.columns)
        self.assertIn('Strategy_Returns', data_with_returns.columns)
        self.assertIn('Cumulative_Returns', data_with_returns.columns)
        # Проверяем, что кумулятивная доходность не содержит NaN после первого периода
        self.assertFalse(data_with_returns['Cumulative_Returns'].isnull().iloc[1:].any())

    def test_generate_recommendation(self):
        """Тестируем функцию generate_recommendation."""
        data_with_bands = calculate_bollinger_bands(self.data.copy())
        recommendation = generate_recommendation(data_with_bands)
        # Проверяем, что рекомендация является строкой
        self.assertIsInstance(recommendation, str)
        # Проверяем, что рекомендация содержит одно из ожидаемых сообщений
        expected_recommendations = [
            "Рекомендация: ПОКУПАТЬ. Цена ниже нижней полосы Боллинджера.",
            "Рекомендация: ПРОДАВАТЬ. Цена выше верхней полосы Боллинджера.",
            "Рекомендация: УДЕРЖИВАТЬ. Цена внутри полос Боллинджера."
        ]
        self.assertIn(recommendation, expected_recommendations)

    def test_full_strategy(self):
        """Полный тест стратегии с отображением графиков и выводом рекомендации."""
        data = self.data.copy()
        data = calculate_bollinger_bands(data)
        data = generate_signals(data)
        data = calculate_returns(data)

        # Отображение графиков
        plot_bollinger_bands(data)
        plot_strategy_performance(data)

        # Получаем и выводим рекомендацию
        recommendation = generate_recommendation(data)
        print(recommendation)

        # Проверяем наличие кумулятивной доходности
        self.assertIn('Cumulative_Returns', data.columns)
        self.assertFalse(data['Cumulative_Returns'].isnull().all())

if __name__ == '__main__':
    unittest.main()