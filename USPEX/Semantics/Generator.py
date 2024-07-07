from abc import ABC, abstractmethod

from ..DataModel.Pool import Pool

class Generator(ABC):

    @abstractmethod
    def call(self, generation: dict[str, Pool]) -> Pool:
        pass
