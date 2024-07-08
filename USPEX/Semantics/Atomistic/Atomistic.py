from abc import ABC, abstractmethod
from pathlib import Path

class Atomistic(ABC):
    @classmethod
    @abstractmethod
    def writeAtomicStructure(cls, filename: Path, system: dict) -> None:
        pass

    @classmethod
    @abstractmethod
    def writeAtomicStructures(cls, filename: Path, systems: list) -> None:
        pass

    @classmethod
    @abstractmethod
    def readAtomicStructure(cls, filename: Path) -> dict:
        pass

    @classmethod
    @abstractmethod
    def readAtomicStructures(cls, filename: Path) -> list:
        pass
