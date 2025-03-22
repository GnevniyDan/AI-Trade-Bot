# Режим работы (минут)
OPERATING_MODE = 10

# Допустимые периоды для индикаторов
DEFAULT_RSI_PERIOD = 14
DEFAULT_STOCH_PERIOD = 14
DEFAULT_BOLLINGER_PERIOD = 20
DEFAULT_MA_SHORT_PERIOD = 9
DEFAULT_MA_LONG_PERIOD = 21

# Каталог хранения данных
DATA_DIR = "storage"

# Классы исключений
# =================

class ApplicationError(Exception):
    """Базовый класс для всех исключений приложения."""
    def __init__(self, message="Ошибка в приложении"):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"

class DatabaseError(ApplicationError):
    """Исключение для ошибок, связанных с базой данных или хранилищем."""
    def __init__(self, message="Ошибка в базе данных"):
        super().__init__(message)

class InvalidInputError(ApplicationError):
    """Исключение для ошибок валидации входных данных."""
    def __init__(self, message="Некорректный ввод данных"):
        super().__init__(message)

class APIError(ApplicationError):
    """Исключение для ошибок при работе с внешними API."""
    def __init__(self, message="Ошибка при обращении к API"):
        super().__init__(message)

class CalculationError(ApplicationError):
    """Исключение для ошибок в вычислениях."""
    def __init__(self, message="Ошибка в расчетах"):
        super().__init__(message)

# Типы данных
# ===========

class todaySupRes:
    """
    Класс для хранения и анализа уровней поддержки и сопротивления.
    
    Атрибуты:
        pivot (float): Уровень точки разворота
        resistance_1, resistance_2, resistance_3 (float): Уровни сопротивления
        support_1, support_2, support_3 (float): Уровни поддержки
    """
    def __init__(self, pivot, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3):
        self.pivot = pivot
        self.resistance_1 = resistance_1
        self.resistance_2 = resistance_2
        self.resistance_3 = resistance_3
        self.support_1 = support_1
        self.support_2 = support_2
        self.support_3 = support_3
        
        # Вычисление средней силы уровней
        self.avg_resistance_gap = (resistance_1 - pivot + resistance_2 - resistance_1 + resistance_3 - resistance_2) / 3
        self.avg_support_gap = (pivot - support_1 + support_1 - support_2 + support_2 - support_3) / 3

    def __repr__(self):
        return (f"Рассчитанные уровни:\n"
                f"Pivot Point: {self.pivot:.2f}\n"
                f"Resistance 1: {self.resistance_1:.2f}\n"
                f"Resistance 2: {self.resistance_2:.2f}\n"
                f"Resistance 3: {self.resistance_3:.2f}\n"
                f"Support 1: {self.support_1:.2f}\n"
                f"Support 2: {self.support_2:.2f}\n"
                f"Support 3: {self.support_3:.2f}")
    
    def is_level_broken(self, price, level_type, level_num):
        """
        Проверяет, был ли пробит указанный уровень.
        
        Аргументы:
            price (float): Текущая цена
            level_type (str): Тип уровня ('resistance' или 'support')
            level_num (int): Номер уровня (1, 2 или 3)
            
        Возвращает:
            bool: True, если уровень пробит, иначе False
        """
        if level_type == 'resistance':
            if level_num == 1:
                return price > self.resistance_1
            elif level_num == 2:
                return price > self.resistance_2
            elif level_num == 3:
                return price > self.resistance_3
        elif level_type == 'support':
            if level_num == 1:
                return price < self.support_1
            elif level_num == 2:
                return price < self.support_2
            elif level_num == 3:
                return price < self.support_3
        return False
    
    def get_current_zone(self, price):
        """
        Определяет, в какой зоне находится текущая цена.
        
        Аргументы:
            price (float): Текущая цена
            
        Возвращает:
            str: Название зоны
        """
        if price > self.resistance_3:
            return "Выше R3 (сильное сопротивление)"
        elif price > self.resistance_2:
            return "Между R2 и R3"
        elif price > self.resistance_1:
            return "Между R1 и R2"
        elif price > self.pivot:
            return "Между Pivot и R1"
        elif price > self.support_1:
            return "Между S1 и Pivot"
        elif price > self.support_2:
            return "Между S2 и S1"
        elif price > self.support_3:
            return "Между S3 и S2"
        else:
            return "Ниже S3 (сильная поддержка)"


class instantCandleReport:
    """
    Класс для хранения отчетов о моментальном состоянии свечи.
    
    Атрибуты:
        RSI (float): Значение индекса относительной силы
        rsipoints (int): Баллы RSI для оценки силы сигнала
    """
    def __init__(self, RSI, rsipoints):
        self.RSI = RSI
        self.rsipoints = rsipoints
    
    def __repr__(self):
        rsi_status = "Перекуплен" if self.RSI > 70 else "Перепродан" if self.RSI < 30 else "Нейтральный"
        return f"RSI: {self.RSI:.2f} ({rsi_status}), Сила сигнала: {self.rsipoints}"


class bollinger:
    """
    Класс для хранения данных индикатора Боллинджера.
    
    Атрибуты:
        recommendation (str): Текстовая рекомендация на основе индикатора
        dataFrame (pandas.DataFrame): DataFrame с рассчитанными значениями индикатора
    """
    def __init__(self, recommendation, dataFrame):
        self.recommendation = recommendation
        self.dataFrame = dataFrame
    
    def __repr__(self):
        return self.recommendation
    
    def get_trend_strength(self):
        """
        Вычисляет силу текущего тренда на основе ширины полос Боллинджера
        
        Возвращает:
            float: Значение от 0 до 1, где 1 - сильный тренд
        """
        if 'Upper' in self.dataFrame.columns and 'Lower' in self.dataFrame.columns:
            last_row = self.dataFrame.iloc[-1]
            band_width = last_row['Upper'] - last_row['Lower']
            return min(band_width / (last_row['Upper'] * 0.1), 1.0)
        return 0.0