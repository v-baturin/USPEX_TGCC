import numpy as np
from collections import Counter
from typing import Dict, Union


class Composition(dict):
    """
    Class describing composition of an atomic structure.
    """

    def __init__(self, composition: Dict[str, int], molecules: Dict[str, Dict[str, int]]):
        """
        Initializes the class.

        :type composition: Dict[str, int]
        :param composition: dictionary describing composition in terms of molecules.
        :type molecules: Dict[str, Dict[str, int]]
        :param molecules: mapping form molecule names to their formulas.
        """
        super().__init__(composition)
        self.molecules = molecules
        self._elementalComposition = {}

    @property
    def elementalComposition(self):
        """
        :rtype: Dict[str, int]
        :return: dictionary describing composition in terms of chemcal elements.
        """
        if not self._elementalComposition:
            comp = Counter()
            for symbol, amount in self.items():
                if symbol in self.molecules:
                    for symbol, value in self.molecules[symbol].items():
                        comp[symbol] += value*amount
                else:
                    comp[symbol] += amount
            self._elementalComposition = dict(comp)
        return self._elementalComposition

class CompositionSpace(object):
    """
    Describes the chemical compositions configuration space.
    """

    def __init__(self, symbols: list, blocks: list=None, range: list=None, minAt: int=None, maxAt: int=None):
        """
        Initializes the class.

        :type blocks: [[...],[...],...]
        :param blocks:
            each system of the configuration space must have composition
            which is span of rows of this parameter; obligatory
        :type range: [[...],[...],...]
        :param range:
            range for each block.
        :type minAt: int
        :param minAt:
            minimum number of atoms or molecules in the unit cell for the first generation.
        :type maxAt: int
        :param maxAt:
            maximum number of atoms or molecules in the unit cell for the first generation.

        """

        self.molecules = {}
        self.moleculesTypeToFormula = {}
        self.symbols = []
        self.chemicalSymbols = []

        chemicalSymbols = []
        for symbol in symbols:
            if isinstance(symbol, dict):
                assert 'molSymbols' in symbol and len(symbol['molSymbols']) == 1
                molSymbol = symbol['molSymbols'][0]
                self.molecules[molSymbol] = symbol
                formula = dict(zip(*np.unique(symbol['symbols'], return_counts=True)))
                self.moleculesTypeToFormula[molSymbol] =  formula
                self.symbols.append(molSymbol)
                chemicalSymbols.extend(symbol['symbols'])
            else:
                self.symbols.append(symbol)
                chemicalSymbols.append(symbol)

        indexes = np.unique(chemicalSymbols, return_index=True)[1]  # alphabetical reordering is not wanted!
        self.chemicalSymbols = [chemicalSymbols[index] for index in sorted(indexes)]

        self.blocks = np.asarray(blocks, dtype=int)
        assert len(self.blocks.shape) == 2 and self.blocks.shape[1] == len(self.symbols)

        if type(minAt) in [int, float] and type(maxAt) in [int, float]:
            assert maxAt >= minAt
            self.minAt = minAt
            self.maxAt = maxAt
        else:
            self.minAt = int(np.sum([np.min(x) * np.sum(block) for block, x in zip(blocks, range)]))
            self.maxAt = int(np.sum([np.max(x) * np.sum(block) for block, x in zip(blocks, range)]))

        self.range = np.array(range, dtype=int)
        assert np.all([len(x) == 2 for x in range]) and len(range) == len(blocks)

        self.isFixedComposition = np.all([x1 == x2 for x1, x2 in self.range])

        self.predefinedCompositions = []
        minBlocks = np.fromiter((minBlocks for minBlocks, maxBlocks in self.range), dtype = int)
        numBlocksArray = np.tile(minBlocks, (len(self.blocks), 1)) + np.diag(np.logical_not(minBlocks))
        for numIons in np.dot(numBlocksArray, self.blocks):
            factor = int(np.ceil(float(self.minAt)/float(numIons.sum())))
            if factor:
                numIons *= factor
            assert numIons.sum() <= self.maxAt
            self.predefinedCompositions.append(Composition(dict(zip(self.symbols, numIons)), self.moleculesTypeToFormula))

    def isGoodComposition(self, composition: Composition) -> bool:
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

    def composition(self, system : dict = None, composition = None):
        return system['structure'].composition if composition is None and system is not None else composition

    def numIons(self, system : dict = None, composition = None):
        """
        Creates numIons array from given composition.

        :type composition: dict
        :param composition: ('element' : 'amount')
        :rtype: list
        :return: list of elements amounts corresponding *symbols* variable of this instance.
        """
        composition = self.composition(system, composition)

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
        :param composition: ('element' : 'amount')
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
                return Composition(dict(zip(self.symbols, numIons)), self.moleculesTypeToFormula)

    def findDesiredComposition(self, composition1: Union[dict, Composition], composition2: Union[dict, Composition],
                               numIons_start: np.ndarray, debug: bool=False):
        """
        Find a composition that requires the least addition/deleting of atoms from child.

        :type composition1: dict
        :param composition1: first parent composition.
        :type composition2: dict
        :param composition2: second parent composition.
        :type numIons_start: numpy.ndarray
        :param numIons_start: starting point for approximation.
        :type debug: bool
        :param debug: False by default. If set to True, use static values instead of random to reproduce results.
        :rtype: tuple of lists
        :return: (numIons, numBlocks) to determine which atoms could be used to make a child.
        """

        maxBlocks = self.numBlocks(composition = composition1) + self.numBlocks(composition = composition2)

        # Initialize outputs:
        numIons = None
        numBlocks = None

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

            try:
                numIons = np.dot(numBlocks, blocks)
            except:
                numIons = None

        return numIons, numBlocks
