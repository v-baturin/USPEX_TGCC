from typing import Callable, Any

from USPEX.DataModel.Flavour import Flavour


class PropertyExtension:

    def __init__(self):
        self.propertyTable: dict[str, Callable[[Any, Flavour], Any]] = {}

    def __call__(self, method: Callable[[Any, Flavour], Any]):
        self.propertyTable[method.__name__] = method
        return method


class ExpressionExtension:

    def __init__(self):
        self.expressionTable: dict[str, Callable] = {}

    def __call__(self, method: Callable):
        self.expressionTable[method.__name__] = method
        return method
