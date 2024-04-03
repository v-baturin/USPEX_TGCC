from abc import ABC, abstractmethod
from typing import Counter

class SymbolsFactoryUtility(ABC):

    @abstractmethod
    def getRandomSymComposition(self, factoryComposition) -> Counter:
        pass
