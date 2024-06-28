from abc import ABC, abstractmethod
from typing import Counter
from .Primitives.Generics import NDArray

class CompositionSpace(ABC):
    """
    Describes the chemical compositions configuration space.
    """

    @abstractmethod
    def isGoodComposition(self, composition) -> bool:
        """
        Method which checks if the structure meets the composition constraints.

        :type composition: dict
        :param composition: composition to be checked.

        :rtype: bool
        :return: True if system meets the constraints, False otherwise.
        """
        pass

    @abstractmethod
    def numIons(self, composition) -> NDArray[int]:
        """
        Creates numIons array from given composition.

        :type composition: dict
        :param composition: (<element> : <amount>)

        :rtype: list
        :return: list of elements amounts corresponding *symbols* variable of this instance.
        """
        pass

    @abstractmethod
    def numBlocks(self, *args, **kwargs) -> NDArray[int]:
        """
        Creates numBlocks array from given composition.

        :type composition: dict
        :param composition: (<element> : <amount>)

        :rtype: list
        :return: list of blocks amounts corresponding *blocks* variable of this instance.
        """
        pass

    @abstractmethod
    def randomComposition(self) -> Counter:
        """
        Creates random numIons array respecting configuration parameters: blocks, range, minAt and maxAt.

        :rtype: list
        :return: list of elements amounts corresponding *symbols* variable of this instance.
        """
        pass

    @abstractmethod
    def findDesiredComposition(self, compositionMax, composition, debug: bool=False) -> Counter:
        """
        Find a composition that requires the least addition/deleting of atoms from child.

        :type compositionMax: dict
        :param compositionMax: composition with maximum possible amounts.
        :type composition: dict
        :param composition: composition with starting point for approximation.
        :type debug: bool
        :param debug: False by default. If set to True, use static values instead of random to reproduce results.

        :rtype: Counter
        :return: found composition.
        """
        pass

    @staticmethod
    @abstractmethod
    def choose(moleculeTypes, desiredComposition) -> NDArray[int]:
        """
        From list of symbols representing molecule types choose only those required by given composition
        and return their indices.

        :param moleculeTypes: list of molecule types to choose from.
        :param desiredComposition: composition with required symbols and amounts.

        :return: indices of symbols in input array which are chosen.
        """
        pass
