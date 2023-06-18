"""
USPEX.Atomistic.CompositionSpace
================================
"""

import logging
import numpy as np
from copy import copy
from collections import Counter

from ..Fitness.CompositionSpaceFunctions import CompositionSpaceFunctions

logger = logging.getLogger(__name__)


class CompositionSpace(object):
    """
    Describes the chemical compositions configuration space.
    """

    fitnessExtension = CompositionSpaceFunctions

    def __init__(self, symbols: list, blocks: list, range: list=None, minAt: int=None, maxAt: int=None):
        """

        :type symbols: [...]
        :param symbols: list of symbols representing elemnts of composition space. Obligatory
        :type blocks: [[...],[...],...]
        :param blocks: list of blocks of elements playing role of basis of composition space. Obligatory
        :type range: [(<min>, <max>),(<min>, <max>),...]
        :param range: list of ranges for each block in composition space. Boundaries are inclusive
        :type minAt: int
        :param minAt:
            minimum number of atoms or molecules in the unit cell.
        :type maxAt: int
        :param maxAt:
            maximum number of atoms or molecules in the unit cell.

        """

        self.symbols = copy(symbols)
        self.nSymbols = len(symbols)
        self.nBlocks = len(blocks)

        self.blocks = np.asarray(blocks, dtype=int)
        assert self.blocks.shape == (self.nBlocks, self.nSymbols)

        if type(minAt) in [int, float] and type(maxAt) in [int, float]:
            assert maxAt >= minAt
            self.minAt = minAt
            self.maxAt = maxAt
            if range is None:
                range = [[0, np.ceil(maxAt / np.sum(block))] for block in self.blocks]
        else:
            if self.nBlocks == 1 and range is None:
                range = [[1, 1]]
            assert range is not None
            self.minAt = int(np.sum([np.min(x) * np.sum(block) for block, x in zip(blocks, range)]))
            self.maxAt = int(np.sum([np.max(x) * np.sum(block) for block, x in zip(blocks, range)]))

        self.range = np.asarray(range, dtype=int)
        assert self.range.shape == (self.nBlocks, 2)

        self.isFixedComposition = np.all([x1 == x2 for x1, x2 in self.range])

        self.predefinedCompositions = []
        minBlocks = np.fromiter((minBlocks for minBlocks, maxBlocks in self.range), dtype = int)
        numBlocksArray = np.tile(minBlocks, (len(self.blocks), 1)) + np.diag(np.logical_not(minBlocks))
        for numIons in np.dot(numBlocksArray, self.blocks):
            factor = int(np.ceil(float(self.minAt)/float(numIons.sum())))
            if factor:
                numIons *= factor
            assert numIons.sum() <= self.maxAt
            self.predefinedCompositions.append(Counter(dict(zip(self.symbols, numIons))))

    def isGoodComposition(self, composition) -> bool:
        """
        Method which checks if the structure meets the composition constraints.

        :type composition: dict
        :param composition: composition to be checked.

        :rtype: bool
        :return: True if system meets the constraints, False otherwise.
        """
        if not set(composition.keys()) <= set(self.symbols):
            return False

        numIons = self.numIons(composition = composition)
        numBlocks = self.numBlocks(composition = composition)

        return np.all(np.dot(numBlocks, self.blocks) == numIons) and \
               np.all(numBlocks >= self.range[:,0]) and \
               np.all(numBlocks <= self.range[:,1]) and \
               self.minAt <= np.sum(numIons) <= self.maxAt

    def numIons(self, composition):
        """
        Creates numIons array from given composition.

        :type composition: dict
        :param composition: (<element> : <amount>)

        :rtype: list
        :return: list of elements amounts corresponding *symbols* variable of this instance.
        """
        num = []
        for symbol in self.symbols:
            if symbol in composition:
                num.append(composition[symbol])
            else:
                num.append(0)
        return np.array(num, dtype=int)

    def numBlocks(self, *args, **kwargs) -> np.ndarray:
        """
        Creates numBlocks array from given composition.

        :type composition: dict
        :param composition: (<element> : <amount>)

        :rtype: list
        :return: list of blocks amounts corresponding *blocks* variable of this instance.
        """
        return np.round(np.linalg.lstsq(self.blocks.T, self.numIons(*args, **kwargs), rcond=None)[0]).astype(int)

    def randomComposition(self):
        """
        Creates random numIons array respecting configuration parameters: blocks, range, minAt and maxAt.

        :rtype: list
        :return: list of elements amounts corresponding *symbols* variable of this instance.
        """
        # First we want use some predefined compositions: pure blocks.
        # And only when we exhaust them we switch to true random.
        while True:
            numBlocks = np.fromiter((np.random.randint(low, high + 1) for low, high in self.range), dtype=int)
            numIons = np.dot(numBlocks, self.blocks)
            if self.minAt <= np.sum(numIons) <= self.maxAt:
                return Counter(dict(zip(self.symbols, numIons)))

    def findDesiredComposition(self, compositionMax,
                               composition, debug: bool=False):
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

        maxBlocks = self.numBlocks(composition = compositionMax)
        numIons_start = np.fromiter((composition[symbol] for symbol in self.symbols), dtype = int)

        maxAtoms = np.dot(maxBlocks, self.blocks)
        maxAdded = maxAtoms - numIons_start  # how many atoms one could possibly add

        blocks = np.asarray(self.blocks, dtype=int)

        if len(blocks.shape) == 1:
            Nb, Nt = 1, blocks.size
            blocks = np.array([blocks])
        else:
            Nb, Nt = blocks.shape  # N of blocks, N of atom types

        tmp = 1  # calculate how many different block combinations we have to check with exhaustive search
        for i in range(Nb):
            min1 = np.max(maxAtoms)
            for j in range(Nt):
                if blocks[i, j] > 0:
                    division = maxAtoms[j] / float(blocks[i, j])
                    if min1 > division:
                        min1 = int(np.floor(division))

            # Now 'min1' says us how many times block 'j' can be fitted into maxAtoms.
            if min1 > 0:
                tmp *= min1

        # Now we do 'greedy' algorithm for big amount of combinations and exhaustive search for small one.
        # 'greedy' algorithm won't give the best answer, but it should be good in most cases.
        # We repeat greedy algorithm 10 (MR: 20?) times and choose the best answer.

        numBlocks = maxBlocks

        if tmp > 0:  # greedy algorithm
            bestGreed = np.sum(maxAtoms)
            for g in range(20):
                # Maximum amount of atoms that could be added to child for any specific type:
                #tolerance = int(round(np.random.rand() * 2.0))
                #if debug:
                #    tolerance = 1

                blockN = np.zeros(Nb, dtype=int)  # specifies the number of blocks in the composition found by algorithm
                composition_tmp = np.copy(numIons_start)
                crutch = np.random.rand(Nb)
                if debug:
                    crutch = np.asarray([0.7947, 0.5449, 0.2])

                blockOrder = crutch.argsort()

                for i in range(Nb):
                    block = blocks[blockOrder[i], :]
                    ind = 1
                    min1 = np.max(maxAtoms)#np.sum(composition_tmp)

                    while True:
                        #if min(composition_tmp - ind * block) < -1 * tolerance:
                        #    break
                        # Take into account maxAtoms, sometimes we can't add atoms at all for varcomp,
                        # since all atoms of specific type could be already in the child.
                        if min(maxAdded + (composition_tmp - ind * block)) < 0:
                            break
                        if min1 >= np.sum(abs(composition_tmp - ind * block)) and \
                                self.range[blockOrder[i], 0] <= ind <= self.range[blockOrder[i], 1]:
                            min1 = np.sum(abs(composition_tmp - ind * block))
                            blockN[blockOrder[i]] = ind
                        ind += 1

                    composition_tmp -= blockN[blockOrder[i]] * block

                if bestGreed > np.sum(abs(numIons_start - np.dot(blockN, blocks))) and \
                                np.all(blockN >= self.range[:,0]) and \
                                np.all(blockN <= self.range[:,1]):
                    bestGreed = np.sum(abs(numIons_start - np.dot(blockN, blocks)))
                    numBlocks = blockN

        return +Counter(dict(zip(self.symbols, np.dot(numBlocks, blocks)))) # We left only positive values in composition

    @staticmethod
    def choose(moleculeTypes, desiredComposition):
        """
        From list of symbols representing molecule types choose only those required by given composition
        and return their indices.

        :param moleculeTypes: list of molecule types to choose from.
        :param desiredComposition: composition with required symbols and amounts.

        :return: indices of symbols in input array which are chosen.
        """
        indices = []
        composition = Counter()
        for i, moleculeType in enumerate(moleculeTypes):
            if composition[moleculeType] < desiredComposition[moleculeType]:
                indices.append(i)
                composition[moleculeType] += 1
        return np.asarray(indices)


