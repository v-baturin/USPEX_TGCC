import numpy as np


class CompositionSpaceFunctions:
    def __init__(self, utility) -> None:
            self.utility = utility

    def numBlocksFromCompositions(self, compositions: np.ndarray):
        """
        For using in **Fitness** infrastructure

        :param compositions: N array of dictionary like compositions.

        :return: N*M array of block numbers, where M number of different blocks defined in this space.
        """
        numBlocks = []
        for composition in compositions:
            numBlocks.append(self.utility.numBlocks(composition))
        return np.asarray(numBlocks)

    def numMolsFromCompositions(self, compositions: np.ndarray):
        """
        For using in **Fitness** infrastructure

        :param compositions: N array of dictionary like compositions.

        :return: N*M array of elements numbers, where M number of different symbols defined in this space.
        """
        numMols = []
        for composition in compositions:
            numMols.append(self.utility.numIons(composition))
        return np.asarray(numMols)
