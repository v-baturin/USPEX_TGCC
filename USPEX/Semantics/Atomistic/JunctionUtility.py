from abc import ABC, abstractmethod

class JunctionUtility(ABC):

    @staticmethod
    @abstractmethod
    def calculateJunctionTypes(structure, junctionsDescription):
        pass
