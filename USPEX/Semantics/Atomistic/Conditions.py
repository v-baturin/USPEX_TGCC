from abc import ABC, abstractmethod

class Conditions(ABC):
    """
    Class describing conditions such as external pressure.
    """

    @abstractmethod
    def putConditions(self, system) -> None:
        """
        Put parameters into system dictionary.

        :param system: dictionary to put parameters into.

        """
        pass
