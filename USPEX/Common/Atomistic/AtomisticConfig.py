'''
@file        Config.py
@author:     Pavel Bushlanov
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        5 December 2016
@brief       Class for description of configuration of target space.
'''


import logging
import numpy as np

from copy import copy
from itertools import combinations_with_replacement, chain
from ase.geometry import get_distances


from ..Config import Config
from .AtomicStructure import AtomicStructure
from .Element import Element
from .Bonds import defaultGoodBonds
from .calcDefaultVolume import calcVolume


# A minimal angle between any two vectors defining the lattice.
_MIN_ANGLE = 55

# A minimal angle between the vector defining the lattice and the
# diagonal of the parallelogram formed by other 2 vectors defining the lattice
_MIN_DIAG_ANGLE = 30
logger = logging.getLogger(__name__)

# Deafult fingerprints tolerance
_DEFAULT_FINGERPRINT_TOLERANCE = 0.008

# Default symmetry tolerance
_DEFAULT_SYMMETRY_TOLERANCE = 0.05


class ChemicalConfig(Config):
    '''
    Class that fix some chemical (more-less) parameters of the system.
    This class does not take into account geometrical or
    '''

    symbols = None
    chemicalSymbols = None
    molecules = None
    volumeType = None
    goodBonds = None
    valences = None
    valenceElectrons = None
    _minVectorLength = None
    externalPressure = None
    minDistMatrice = None
    CenterminDistMatrice = None

    def __init__(self, symbols : list, volumeType : str=None, ionDistances : dict=None, goodBonds : list=None,
                 valences : list=None, minVectorLength : int=None, valenceElectrons : list=None,
                 externalPressure : float=0.0001, moleculesDistinctCheck = True, MolCenters : dict=None, **kwargs):
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
            if self.molecules:
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

        self.moleculesDistinctCheck = moleculesDistinctCheck

    def toDICT(self):
        '''
        Method which creates dictionary representation of the config.

        :return: Dictionary representing the config.
        '''
        dct = copy(self.__dict__)
        dct['goodBonds'] = dct['goodBonds'].tolist()
        dct['valences'] = dct['valences'].tolist()
        dct['valenceElectrons'] = dct['valenceElectrons'].tolist()
        dct['minDistMatrice'] = dct['minDistMatrice'].tolist()
        dct['CenterminDistMatrice'] = dct['CenterminDistMatrice'].tolist()
        return dct

    @classmethod
    def fromDICT(cls, dct : dict):
        '''
        Method which reconstructs ChemicalConfig from dictionary representation.

        :param dct: Dictionary representing the config.
        '''
        dct['goodBonds'] = np.asarray(dct['goodBonds'])
        dct['valences'] = np.asarray(dct['valences'])
        dct['valenceElectrons'] = np.asarray(dct['valenceElectrons'])
        dct['minDistMatrice'] = np.asarray(dct['minDistMatrice'])
        dct['CenterminDistMatrice'] = np.asarray(dct['CenterminDistMatrice'])
        config = cls(symbols=[])
        config.__dict__ = copy(dct)
        return config

    def minVectorLength(self, system : AtomicStructure) -> float:
        '''
        Here we check whether minVectorLength already specified in the input.
        If not, we calculate it based on the chemical compound of current system (see. Manual)

        :param system:
        :return:
        '''
        if self._minVectorLength is None:
            radii = np.fromiter((Element(symbol).covalent_radius for symbol in system.chemicalSymbols), dtype=float)
            return 1.8 * radii.max()
        else:
            return self._minVectorLength

    def isGoodDistances(self, system : AtomicStructure) -> bool:
        '''
        Method which checks if the structure meet minimal distance constraint.

        :param system:
        :return:
        '''
        return system.isGoodDistances(self.chemicalSymbols, self.minDistMatrice)

    def isGoodCenterDistances(self, molSymbols, coordinates, cell, pbc=True) -> bool:
        '''
        Method which checks if the structure meet minimal molecular center distance constraint.

        :param system:
        :param molSymbols:
        :return:
        '''
        indices = np.fromiter((self.symbols.index(symbol) for symbol in molSymbols), dtype=int)
        cmDM = self.CenterminDistMatrice[tuple(np.meshgrid(indices, indices))]
        cmDM -= np.diag(np.diag(cmDM))
        D, D_len = get_distances(np.dot(coordinates, cell), cell=cell, pbc=pbc)
        return np.all(D_len >= cmDM.T)

    def isGoodLattice(self, system : AtomicStructure) -> bool:
        '''
        This function checks whether the given lattice fulfills the hard constraints for lattices.
        There are three sorts of hard constraints.
        1) A minimal angle between any two vectors defining the lattice.
        2) A minimal distance between any two planes; was lately changed to simply lattice vector length
        3) A minimal angle between the vector defining the lattice and the
           diagonal of the parallelogram formed by other 2 vectors defining the lattice

        :param system: system that will be checked
        :return lat_OK: boolean value if the lattice is fine for further processing.
        '''

        Lattice = system.cell
        angLattice = system.get_cell_lengths_and_angles()
        lat_OK = True

        # In this function both the matrix and the parameter representation is required, so first we prepare the two types.

        #        # Ensure we deal with NumPy arrays:
        #        if type(Lattice) != np.ndarray:
        #            Lattice = np.asarray(Lattice)

        #        if Lattice.shape[0] == 3:  # 3x3 case -> convert to 1x6 to get angles:
        #            angLattice = np.asarray(latConverter(np.ndarray.tolist(Lattice)))
        #        else:  # 1x6 case -> convert to 3x3 to get lattice:
        #            angLattice = Lattice
        #            Lattice = np.asarray(latConverter(np.ndarray.tolist(Lattice)))

        # Calculate the angles is degrees:
        angles = copy(angLattice[3:6])  # * 180. / np.pi
        angLattice[3:6] = angles / 180.0 * np.pi

        # The following is constraint 1)
        # check whether none of the angles is two small. Note that the problem is
        # symmetric around 90 degrees, that's the reason for 180-minAngle.
        if np.where(angles < _MIN_ANGLE)[0].shape[0] != 0 or \
                np.where(angles > (180.0 - _MIN_ANGLE))[0].shape[0] != 0:
            lat_OK = False

        # The following is constraint 2)
        # We receive the distance by dividing the volume (found by det(lattice)) by the area of any two vectors

        vol = abs(np.linalg.det(Lattice))

        # We don't need dist variable anymore since the corresponding Matlab condition is commented:
        # %if ~isempty(find(dist<minVectorLength))
        '''
        dist = np.zeros(3)
        dist[0] = vol/(angLattice[0, 0] * angLattice[0, 1] * sin(angLattice[0, 5]))
        dist[1] = vol/(angLattice[0, 0] * angLattice[0, 2] * sin(angLattice[0, 4]))
        dist[2] = vol/(angLattice[0, 1] * angLattice[0, 2] * sin(angLattice[0, 3]))
        '''

        if np.where(angLattice[0:3] < self.minVectorLength(system))[0].shape[0] > 0:
            lat_OK = False

        if np.where(np.isreal(Lattice) == False)[0].shape[0] > 0:
            lat_OK = False

        # The following is constraint 3)
        # check whether none of the angles between the vector defining the lattice and the
        # diagonal of the parallelogram formed by other 2 vectors defining the
        # lattice is two small.

        if lat_OK:  # if it's not 0 it means there are no 0-length vectors
            a_bc = Lattice[0, 0] * (Lattice[1, 0] + Lattice[2, 0]) + Lattice[0, 1] * (Lattice[1, 1] + Lattice[2, 1]) + \
                   Lattice[0, 2] * (Lattice[1, 2] + Lattice[2, 2])
            ab_c = Lattice[2, 0] * (Lattice[1, 0] + Lattice[0, 0]) + Lattice[2, 1] * (Lattice[1, 1] + Lattice[0, 1]) + \
                   Lattice[2, 2] * (Lattice[1, 2] + Lattice[0, 2])
            b_ca = Lattice[1, 0] * (Lattice[0, 0] + Lattice[2, 0]) + Lattice[1, 1] * (Lattice[0, 1] + Lattice[2, 1]) + \
                   Lattice[1, 2] * (Lattice[0, 2] + Lattice[2, 2])

            # |b+c|^2=|b|^2+|c|^2+2|b||c|cos(bc):
            anglesDiag = np.zeros(3)
            anglesDiag[0] = np.arccos(a_bc / (angLattice[0] * np.sqrt(
                angLattice[1] ** 2 + angLattice[2] ** 2 + 2. * angLattice[1] * angLattice[2] * np.cos(
                    angLattice[3]))))
            anglesDiag[1] = np.arccos(b_ca / (angLattice[1] * np.sqrt(
                angLattice[0] ** 2 + angLattice[2] ** 2 + 2. * angLattice[0] * angLattice[2] * np.cos(
                    angLattice[4]))))
            anglesDiag[2] = np.arccos(ab_c / (angLattice[2] * np.sqrt(
                angLattice[1] ** 2 + angLattice[0] ** 2 + 2. * angLattice[1] * angLattice[0] * np.cos(
                    angLattice[5]))))
            anglesDiag = anglesDiag * 180. / np.pi

            if np.where(anglesDiag < _MIN_DIAG_ANGLE)[0].shape[0] != 0 or \
                    np.where(anglesDiag > (180.0 - _MIN_DIAG_ANGLE))[0].shape[0] != 0:
                lat_OK = False

        return lat_OK

    def isGoodSystem(self, system : AtomicStructure) -> bool:
        '''
        Method which checks if the structure meet constraints.

        :param system:
        :return:
        '''
        isGoodSystem = self.isGoodDistances(system)
        if isGoodSystem and self.moleculesDistinctCheck:
            isGoodSystem = system.isMoleculesDistinct()
        return isGoodSystem

    def calcVolume(self):
        '''
        This function estimates volume occupied by set of atoms described by chemical formula.

        :return: list of corresponding volumes
        '''
        volume = []
        for symbol in self.symbols:
            if symbol not in self.molecules:
                volume.append(calcVolume(self.externalPressure, symbol, self.volumeType))
            else:
                volume.append(sum(calcVolume(self.externalPressure, s, self.volumeType)
                                  for s in self.molecules[symbol]['symbols']))
        return np.array(volume)


class AtomisticConfig(ChemicalConfig):
    '''
    Structure with more-less most important parameters for the calculation that are set from input of the calculation.
    '''

    blocks = None
    fixed = None
    minAt = None
    maxAt = None
    isFixedComposition = None
    fingerprints = None

    def __init__(self, blocks=None, fixed=None, minAt=None, maxAt=None, fingerprints : dict=None,
                 magRatio=None, magSymm=None, sym_tolerance=None, **kwargs):
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
                # use magnetic moments in the MOL file, if present, otherwise set them to zero
                magSymm[symbol] = [0] if 'MOL' in symbol else int(Element(symbol).z in default)
        self.magSymm = magSymm

        if magRatio is not None:
            assert isinstance(magRatio, list) and sum(magRatio) > 0
            if not np.isclose(sum(magRatio), 1.0):
                magRatio = np.array(magRatio) / sum(magRatio)
                logger.info('magRatio has been rescaled.')
            self.magRatio = np.array(magRatio)
        else:
            self.magRatio = np.array([1, 0, 0, 0, 0, 0, 0])

        self.fingerprints = copy(fingerprints) if fingerprints is not None else {}
        if 'tolerance' not in self.fingerprints:
            self.fingerprints['tolerance'] = _DEFAULT_FINGERPRINT_TOLERANCE

        if sym_tolerance is not None:
            if isinstance(sym_tolerance, str):
                if 'high' in sym_tolerance:
                    self.sym_tolerance = 0.05
                elif 'medium' in sym_tolerance:
                    self.sym_tolerance = 0.1
                elif 'low' in sym_tolerance:
                    self.sym_tolerance = 0.2
                else:
                    self.sym_tolerance = _DEFAULT_SYMMETRY_TOLERANCE
            elif isinstance(sym_tolerance, (float, int)):
                self.sym_tolerance = float(sym_tolerance)
            else:
                self.sym_tolerance = _DEFAULT_SYMMETRY_TOLERANCE
        else:
            self.sym_tolerance = _DEFAULT_SYMMETRY_TOLERANCE

    def toDICT(self):
        '''
        Method which creates dictionary representation of the config.

        :return: Dictionary representing the config.
        '''
        dct = super().toDICT()
        dct['blocks'] = dct['blocks'].tolist()
        dct['fixed'] = dct['fixed'].tolist()
        dct['magRatio'] = dct['magRatio'].tolist()
        return dct

    @classmethod
    def fromDICT(cls, dct : dict):
        '''
        Method which reconstructs AtomisticConfig from dictionary representation.

        :param dct: Dictionary representing the config.
        '''
        dct['blocks'] = np.asarray(dct['blocks'])
        dct['fixed'] = np.asarray(dct['fixed'])
        dct['magRatio'] = np.asarray(dct['magRatio'])
        return super().fromDICT(dct)

    def isGoodComposition(self, system : AtomicStructure) -> bool:
        '''
        Method which checks if the structure meet composition constraint.

        :param system:
        :return:
        '''
        if not set(system.composition.keys()) <= set(self.symbols):
            return False
        numIons = self.numIons(system.composition)
        numBlocks = self.numBlocks(system.composition)
        return np.all(np.dot(numBlocks, self.blocks) == numIons) and \
               np.all(numBlocks >= self.fixed[:,0]) and \
               np.all(numBlocks <= self.fixed[:,1]) and \
               np.sum(numIons) >= self.minAt and \
               np.sum(numIons) <= self.maxAt

    def numIons(self, composition):
        '''
        Creates numIons array from given composition.
        :param composition:
        :return:
        '''
        num = []
        for symbol in self.symbols:
            if symbol in composition:
                num.append(composition[symbol])
            else:
                num.append(0)
        return np.array(num, dtype=int)

    def numBlocks(self, composition) -> np.ndarray:
        '''
        Creates numIons array from given composition.

        :param composition:
        :return:
        '''
        return np.round(np.linalg.lstsq(self.blocks.T, self.numIons(composition), rcond=None)[0]).astype(int)

    def randomComposition(self):
        while True:
            numBlocks = np.fromiter((np.random.randint(low, high + 1) for low, high in self.fixed), dtype=int)
            numIons = np.dot(numBlocks, self.blocks)
            if np.sum(numIons) >= self.minAt and np.sum(numIons) <= self.maxAt:
                return numIons

    def isGoodSystem(self, system : AtomicStructure) -> bool:
        '''
        Method which checks if the structure meet constraints.

        :param system:
        :return:
        '''
        return self.isGoodComposition(system) and super(AtomisticConfig, self).isGoodSystem(system)

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
