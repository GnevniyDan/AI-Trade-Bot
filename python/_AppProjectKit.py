"""
Библиотека наших классов ошибок и типов данных
"""

#Режим работы (минут)
OPERATING_MODE = 10

#Ошибки
class ApplicationError(Exception):
    pass

class DatabaseError(ApplicationError):
    pass

class InvalidInputError(ApplicationError):
    def __init__(self, message="Некорректный ввод данных"):
        self.message = message
        super().__init__(self.message)


#Типы данных
class todaySupRes:
    def __init__(self, pivot, resistance_1, resistance_2, resistance_3, support_1, support_2, support_3):
        self.pivot = pivot
        self.resistance_1 = resistance_1
        self.resistance_2 = resistance_2
        self.resistance_3 = resistance_3
        self.support_1 = support_1
        self.support_2 = support_2
        self.support_3 = support_3

    def __repr__(self):
        return f"Рассчитанные уровни:\nPivot Point: {self.pivot}\nResistance 1: {self.resistance_1}\nResistance 2: {self.resistance_2}\nResistance 3: {self.resistance_3}\nSupport 1: {self.support_1}\nSupport 2: {self.support_2}\nSupport 3: {self.support_3}"
    

class instantCandleReport:
    def __init__(self, RSI: float, rsipoints: int, ):
        self.RSI = RSI
        self.rsipoints = rsipoints


class bollinger:
    def __init__(self, recommendation, dataFrame):
        self.recommendation = recommendation
        self.dataFrame = dataFrame