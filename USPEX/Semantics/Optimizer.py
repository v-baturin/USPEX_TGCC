from abc import ABC, abstractmethod

from ..DataModel.Pool import Pool

class Optimizer(ABC):

    @abstractmethod
    async def update(self, population: Pool, parentsGeneration: dict[str, Pool]) -> tuple[dict[str, Pool], bool, bool]:
        pass
