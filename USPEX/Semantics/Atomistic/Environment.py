from abc import ABC, abstractmethod
from .Primitives.AtomicStructure import AtomicStructure

class Environment(ABC):
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """

    @abstractmethod
    def assemble(self, molecules, cell, **kwargs) -> list[tuple[AtomicStructure, list[int]]]:
        pass

    @staticmethod
    @abstractmethod
    def build(*args, **kwargs) -> 'Environment':
        """
        Builds the environment objects for a given description.
        """
        pass
