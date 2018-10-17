import logging
import numpy as np

from copy import copy
from itertools import combinations_with_replacement, chain

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from USPEX.Common.Atomistic.Element import Element
from USPEX.Common.Atomistic.AtomisticConfig import ChemicalConfig, AtomisticConfig

from .Bonds import defaultGoodBonds
from .calcDefaultVolume import calcVolume


class ChemicalConfigPrivate(ChemicalConfig):
    '''
    Class that fix some chemical (more-less) parameters of the system.
    This class does not take into account geometrical or
    '''

    def __init__(self, symbols : list, volumeType : str=None, ionDistances : dict=None, goodBonds : list=None,
                 valences : list=None, minVectorLength : int=None, valenceElectrons : list=None,
                 externalPressure : float=0.0001, MolCenters : dict=None, **kwargs):
        '''

        :param symbols:
        :param volumeType:
        :param ionDistances:
        :param goodBonds:
        :param valences:
        :param valenceElectrons:
        :param externalPressure:
        :param MolCenters:
        :param kwargs:
        '''

        super().__init__(**kwargs)
        if symbols is None:
            return

        self.molecules = {}
        self.symbols = []
        self.chemicalSymbols = []

        chemicalSymbols = []
        for symbol in symbols:
            if isinstance(symbol, dict):
                assert 'molSymbols' in symbol and len(symbol['molSymbols']) == 1
                molSymbol = symbol['molSymbols'][0]
                self.molecules[molSymbol] = symbol
                self.symbols.append(molSymbol)
                chemicalSymbols.extend(symbol['symbols'])
            else:
                self.symbols.append(symbol)
                chemicalSymbols.append(symbol)

        indexes = np.unique(chemicalSymbols, return_index=True)[1]  # alphabetical reordering is not wanted!
        self.chemicalSymbols = [chemicalSymbols[index] for index in sorted(indexes)]

        if volumeType is None:
            if len(self.symbols) != len(self.chemicalSymbols):
                self.volumeType = 'mol'
            else:
                self.volumeType = 'atom'
        else:
            self.volumeType = volumeType


        if goodBonds is not None:
            self.goodBonds = np.asarray(goodBonds)
            assert len(self.goodBonds.shape) == 2
        else:
            self.goodBonds = defaultGoodBonds(self.chemicalSymbols)

        if valences is not None:
            self.valences = np.asarray(valences)
        else:
            self.valences = np.asarray([Element(symbol).valence for symbol in self.chemicalSymbols])

        if valenceElectrons is not None:
            self.valenceElectrons = np.asarray(valenceElectrons)
        else:
            self.valenceElectrons = np.asarray([Element(symbol).valence_electrons for symbol in self.chemicalSymbols], dtype=int)

        self._minVectorLength = minVectorLength if isinstance(minVectorLength, float) and minVectorLength > 0.0 else None

        assert externalPressure >= 0.0
        self.externalPressure = externalPressure

        self.minDistMatrice = np.zeros((len(self.chemicalSymbols), len(self.chemicalSymbols)))
        if not ionDistances:
            radii = [calcVolume(self.externalPressure, symbol, self.volumeType) ** (1.0 / 3.0) for symbol in self.chemicalSymbols]
            for i, j in combinations_with_replacement(range(len(self.chemicalSymbols)), 2):
                if self.volumeType != 'mol':
                    self.minDistMatrice[i, j] = self.minDistMatrice[j, i] = min(0.22 * (radii[i] + radii[j]), 1.2)
                else:
                    self.minDistMatrice[i, j] = self.minDistMatrice[j, i] = 0.45 * (radii[i] + radii[j])
        else:
            for i, j in combinations_with_replacement(range(len(self.chemicalSymbols)), 2):
                arg = '{}-{}'.format(self.chemicalSymbols[i],self.chemicalSymbols[j])
                self.minDistMatrice[i,j] = self.minDistMatrice[j,i] = ionDistances[arg]


        self.CenterminDistMatrice = np.zeros((len(self.symbols), len(self.symbols)))
        radii = []
        for s in self.symbols:
            if s not in self.molecules:
                radii.append(0.22*calcVolume(self.externalPressure, s, self.volumeType) ** (1.0 / 3.0))
            else:
                molecule = AtomicStructure.fromDICT(self.molecules[s])
                molecule.set_masses([1] * len(molecule))
                molecule.translate(-molecule.get_center_of_mass())
                values, vectors = molecule.get_moments_of_inertia(vectors=True)
                ind = np.argsort(values)[0]
                short_direction = vectors[ind]
                height_map = [np.abs(np.dot(pos, short_direction)) for pos in molecule.get_positions()]
                ind = np.argsort(height_map)[0]
                s = molecule.get_chemical_symbols()[ind]
                radii.append(0.45 * np.power(calcVolume(self.externalPressure, s, self.volumeType), 1 / 3.0) + height_map[ind])
        for i,j in combinations_with_replacement(range(len(radii)),2):
            self.CenterminDistMatrice[i, j] = self.CenterminDistMatrice[j, i] = (radii[i] + radii[j])

    def calcVolume(self):
        '''
        This function estimates volume occupied by set of atoms described by chemical formula.
        return: list of corresponding volumes
        '''
        volume = []
        for symbol in self.symbols:
            if symbol not in self.molecules:
                volume.append(calcVolume(self.externalPressure, symbol, self.volumeType))
            else:
                volume.append(sum(calcVolume(self.externalPressure, s, self.volumeType)
                                  for s in self.molecules[symbol]['symbols']))
        return np.array(volume)


class AtomisticConfigPrivate(AtomisticConfig, ChemicalConfigPrivate):
    '''
    Structure with more-less most important parameters for the calculation that are set from input of the calculation.
    '''

    def __init__(self, blocks=None, fixed=None, minAt=None, maxAt=None, fingerprints : dict = None,
                 magRatio=None, magSymm=None, **kwargs):
        '''
        :param symbols: [formula1, formula2, ...]
                        where formula* is str - list of chemical formulas
                        of molecules constituting the system; obligatory
        :param blocks: [...],[...],...] after conversion 2D numpy array of ints, each system of the configuration space
                                        must have composition which is span of rows of this parameter; obligatory
        :param fixed:
        :param magRatio: fraction of structures which are produced for each magnetic type
        :param magSymm: lists the atomic types whose spin value will be set according to symmetries in the unit cell
        :param kwargs:
        '''

        super().__init__(**kwargs)
        if blocks is None and fixed is None:
            return


        self.blocks = np.array(blocks, dtype=int)
        assert len(self.blocks.shape) == 2 and self.blocks.shape[1] == len(self.symbols)

        if type(minAt) in [int, float] and type(maxAt) in [int, float]:
            assert maxAt >= minAt
            self.minAt = minAt
            self.maxAt = maxAt
        else:
            self.minAt = int(np.sum([np.min(x) * np.sum(block) for block, x in zip(blocks, fixed)]))
            self.maxAt = int(np.sum([np.max(x) * np.sum(block) for block, x in zip(blocks, fixed)]))

        self.fixed = np.array(fixed, dtype=int)
        assert np.all([len(x) == 2 for x in fixed]) and len(fixed) == len(blocks)

        self.isFixedComposition = bool(np.all([x[0] == x[1] for x in self.fixed]))

        if magSymm is not None:
            assert isinstance(magSymm, dict)
        else:
            magSymm = {}
            default = [i for i in chain(range(21, 31), range(39, 49))]   #first 2 rows of transition metals
            for symbol in self.symbols:
                if 'MOL' in symbol:
                    magSymm[symbol] = [0] # use magnetic moments in the MOL file, if present, otherwise set them to zero
                else:
                    magSymm[symbol] = int(Element(symbol).z in default)
        self.magSymm = magSymm

        if magRatio is not None:
            assert isinstance(magRatio, list) and sum(magRatio) > 0
            if not np.isclose(sum(magRatio), 1.0):
                magRatio = np.array(magRatio) / sum(magRatio)
                logging.info('magRatio has been rescaled.')
            self.magRatio = np.array(magRatio)
        else:
            self.magRatio = np.array([1, 0, 0, 0, 0, 0, 0])

        if fingerprints is not None:
            self.fingerprints = copy(fingerprints)
        else:
            self.fingerprints = {}

        if 'tolerance' not in self.fingerprints:
            self.fingerprints['tolerance'] = 0.008

    def findDesiredComposition(self, system1 : AtomicStructure, system2 : AtomicStructure, composition, debug=False):
        """
        How to find a composition that requires the least addition/deleting of atoms from child.
        One should be able to compose numIons out of blocks.
        :param system1:
        :param system2:
        :param composition:
        :param debug: False by default. If set to True, use static values instead of random to reproduce results.
        :return numIons:
        :return numBlocks: full amount of blocks from both parents, determines which atoms could be used to make a child.
        """

        maxBlocks = self.numBlocks(system1.composition) + self.numBlocks(system2.composition)

        # Initialize outputs:
        numIons = None
        numBlocks = None

        maxAtoms = np.dot(maxBlocks, self.blocks)
        maxAdded = maxAtoms - composition  # how many atoms one could possibly add

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
                composition_tmp = np.copy(composition)
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
                        # Take into account maxAtoms, sometimes we can't add atoms at all for varcomp, since all atoms of
                        # specific type could be already in the child.
                        if min(maxAdded + (composition_tmp - ind * block)) < 0:
                            break
                        if min1 >= np.sum(abs(composition_tmp - ind * block)) and \
                                ind >= self.fixed[blockOrder[i],0] and \
                                ind <= self.fixed[blockOrder[i],1]:
                            min1 = np.sum(abs(composition_tmp - ind * block))
                            blockN[blockOrder[i]] = ind
                        ind += 1

                    composition_tmp -= blockN[blockOrder[i]] * block

                if bestGreed > np.sum(abs(composition - np.dot(blockN, blocks))) and \
                                np.all(blockN >= self.fixed[:,0]) and \
                                np.all(blockN <= self.fixed[:,1]):
                    bestGreed = np.sum(abs(composition - np.dot(blockN, blocks)))
                    numBlocks = blockN

            try:
                numIons = np.dot(numBlocks, blocks)
            except:
                numIons = None

        return numIons, numBlocks
