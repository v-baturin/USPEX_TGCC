"""
USPEX.Common.Atomistic.AtomisticConfig
======================================

Description of configuration of target space

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np

from ase.geometry import get_distances
from copy import copy
from itertools import combinations_with_replacement, chain
from typing import Dict, List


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

# Default fingerprint tolerance
_DEFAULT_FINGERPRINT_TOLERANCE = 0.008

# Default symmetry tolerance
_DEFAULT_SYMMETRY_TOLERANCE = 0.05


class ChemicalConfig(Config):
    """
    Class that fixes some chemical (more or less) parameters of the system.
    """

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

    def __init__(self, symbols: list, volumeType: str=None, ionDistances: dict=None, goodBonds: dict=None,
                 valences: Dict[str, float]=None, minVectorLength: float=None, valenceElectrons: Dict[str, float]=None,
                 externalPressure: float=0.0001, moleculesDistinctCheck: bool=True, molCenters: list=None, **kwargs):
        """
        :type symbols: list[str] or list[dict]
        :param symbols:
            list of element symbols or dictionaries describing molecules allowed for this configuration space.
        :type volumeType: str
        :param volumeType:
            'atom' or 'mol', one of the two possible environments for
            volume estimation. The 'mol' environment is less dense.
        :type ionDistances: dict
        :param ionDistances:
            dictionary describing minimal interatomic distances.
            example {'C_C': 1.0, 'C_H': 0.8, , 'H_C': 0.8, 'H_H': 0.5}
        :type goodBonds: list
        :param goodBonds:
            specifies, in a square matrix form, the minimum bond valences
            for contacts that will be considered as important bonds.
        :type valences: list
        :param valences:
            describes the valences of each type of atom.
        :type minVectorLength: float
        :param minVectorLength:
            sets the minimum length of a cell parameter of a newly generated structure.
        :type valenceElectrons: list
        :param valenceElectrons:
            number of valence electrons for each type of atoms.
        :type externalPressure: float
        :param externalPressure:
            external pressure in GPa.
        :type moleculesDistinctCheck: bool
        :param moleculesDistinctCheck:
            if True, check if molecules do not interpenetrate each other.
        :type molCenters: list
        :param molCenters:
            matrix of minimal distances between the geometric centers of molecules.
        :type kwargs: dict
        :param kwargs:
            additional arguments and keywords used to initialize the parent class Config.
        """

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

        self.volumeType = ('mol' if self.molecules else 'atom') if not volumeType else volumeType

        if goodBonds is not None:
            self.goodBonds = goodBonds
        else:
            self.goodBonds = {}
            goodBonds = defaultGoodBonds(self.chemicalSymbols)
            for i, j in combinations_with_replacement(range(len(self.chemicalSymbols)), 2):
                arg = '{}-{}'.format(self.chemicalSymbols[i],self.chemicalSymbols[j])
                self.goodBonds[arg] = goodBonds[i,j]
                arg = '{}-{}'.format(self.chemicalSymbols[j], self.chemicalSymbols[i])
                self.goodBonds[arg] = goodBonds[j, i]

        self.valences = valences if valences is not None else {symbol: Element(symbol).valence
                                                               for symbol in self.chemicalSymbols}
        self.valenceElectrons = valenceElectrons if valenceElectrons is not None else {symbol: Element(symbol).valence_electrons
                                                                                       for symbol in self.chemicalSymbols}

        self._minVectorLength = minVectorLength if isinstance(minVectorLength, float) and minVectorLength > 0 else None

        assert externalPressure >= 0
        self.externalPressure = externalPressure

        self.minDistMatrice = np.zeros((len(self.chemicalSymbols), len(self.chemicalSymbols)))
        if not ionDistances:
            radii = [calcVolume(self.externalPressure, symbol, self.volumeType) ** (1.0 / 3.0)
                     for symbol in self.chemicalSymbols]
            for i, j in combinations_with_replacement(range(len(self.chemicalSymbols)), 2):
                if self.volumeType != 'mol':
                    self.minDistMatrice[i, j] = self.minDistMatrice[j, i] = min(0.22 * (radii[i] + radii[j]), 1.2)
                else:
                    self.minDistMatrice[i, j] = self.minDistMatrice[j, i] = 0.45 * (radii[i] + radii[j])
        else:
            for i, j in combinations_with_replacement(range(len(self.chemicalSymbols)), 2):
                arg = '{}-{}'.format(self.chemicalSymbols[i], self.chemicalSymbols[j])
                self.minDistMatrice[i, j] = self.minDistMatrice[j, i] = ionDistances[arg]

        if molCenters:
            self.CenterminDistMatrice = np.asarray(molCenters, dtype=float)
        else:
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
                    radii.append(0.45 * np.power(calcVolume(self.externalPressure, s, self.volumeType), 1 / 3.0)
                                 + height_map[ind])
            for i, j in combinations_with_replacement(range(len(radii)), 2):
                self.CenterminDistMatrice[i, j] = self.CenterminDistMatrice[j, i] = (radii[i] + radii[j])

        self.moleculesDistinctCheck = moleculesDistinctCheck

    def toDICT(self):
        """
        Method which creates a dictionary representation of the config.

        :rtype: dict
        :return: dictionary representing the config.
        """
        dct = copy(self.__dict__)
        dct['goodBonds'] = dct['goodBonds'].tolist()
        dct['valences'] = dct['valences'].tolist()
        dct['valenceElectrons'] = dct['valenceElectrons'].tolist()
        dct['minDistMatrice'] = dct['minDistMatrice'].tolist()
        dct['CenterminDistMatrice'] = dct['CenterminDistMatrice'].tolist()
        return dct

    @classmethod
    def fromDICT(cls, dct: dict):
        """
        Method which reconstructs ChemicalConfig from dictionary representation.

        :type dct: dict
        :param dct: dictionary representing the config.
        """
        dct['goodBonds'] = np.asarray(dct['goodBonds'])
        dct['valences'] = np.asarray(dct['valences'])
        dct['valenceElectrons'] = np.asarray(dct['valenceElectrons'])
        dct['minDistMatrice'] = np.asarray(dct['minDistMatrice'])
        dct['CenterminDistMatrice'] = np.asarray(dct['CenterminDistMatrice'])
        config = cls(symbols=[])
        config.__dict__ = copy(dct)
        return config

    def minVectorLength(self, system: AtomicStructure) -> float:
        """
        Here we check whether minVectorLength was already specified in the input.
        If not, we calculate it based on the chemical composition of the current system (see manual).

        :type system: :class:`AtomicStructure`
        :param system: system for which we will determine minimal vector length.
        :rtype: float
        :return: minimal vector length.
        """
        if self._minVectorLength is None:
            radii = np.fromiter((Element(symbol).covalent_radius for symbol in system.chemicalSymbols), dtype=float)
            return 1.8 * radii.max()
        else:
            return self._minVectorLength

    def isGoodDistances(self, system: AtomicStructure) -> bool:
        """
        Method which checks if the structure meets the minimal distance constraints.

        :type system: :class:`AtomicStructure`
        :param system: system to be checked.
        :rtype: bool
        :return: True if system meets the constraints, False otherwise.
        """
        return system.isGoodDistances(self.chemicalSymbols, self.minDistMatrice)

    def isGoodCenterDistances(self, molSymbols: list, coordinates: list, cell: list, pbc=True) -> bool:
        """
        Method which checks if the structure meets the minimal molecular center distance constraints.

        :type molSymbols: list
        :param molSymbols: list of molecular symbols.
        :type coordinates: list
        :param coordinates: list of corresponding coordinates of molecular centers.
        :type cell: list
        :param cell: unit cell parameters.
        :type pbc: bool or list[bool]
        :param pbc:
            a value of True would give periodic boundary conditions along all three axes. It is possible
            to give a sequence of three booleans to specify periodicity along specific axes.
        :rtype: bool
        :return: True if system meets the constraints, False otherwise.
        """
        indices = np.fromiter((self.symbols.index(symbol) for symbol in molSymbols), dtype=int)
        cmDM = self.CenterminDistMatrice[tuple(np.meshgrid(indices, indices))]
        cmDM -= np.diag(np.diag(cmDM))
        D, D_len = get_distances(np.dot(coordinates, cell), cell=cell, pbc=pbc)
        return np.all(D_len >= cmDM.T)

    def isGoodLattice(self, system: AtomicStructure) -> bool:
        """
        This function checks whether the given lattice fulfils the hard constraints for lattices.
        There are three sorts of hard constraints:
        1) A minimal angle between any two vectors defining the lattice.
        2) A minimal distance between any two planes; was lately changed to simply lattice vector length
        3) A minimal angle between the vector defining the lattice and the
        diagonal of the parallelogram formed by other 2 vectors defining the lattice

        :type system: :class:`AtomicStructure`
        :param system: system that will be checked.
        :rtype: bool
        :return: True if the lattice is fine, False otherwise.
        """

        lattice = system.cell
        angLattice = system.get_cell_lengths_and_angles()
        lat_OK = True

        # In this function both the matrix and the parameter representation are required,
        # so first we prepare the two types.

        #        # Ensure we deal with NumPy arrays:
        #        if type(Lattice) != np.ndarray:
        #            Lattice = np.asarray(Lattice)

        #        if Lattice.shape[0] == 3:  # 3x3 case -> convert to 1x6 to get angles:
        #            angLattice = np.asarray(latConverter(np.ndarray.tolist(Lattice)))
        #        else:  # 1x6 case -> convert to 3x3 to get lattice:
        #            angLattice = Lattice
        #            Lattice = np.asarray(latConverter(np.ndarray.tolist(Lattice)))

        # Calculate the angles in radians:
        angles = copy(angLattice[3:6])  # * 180. / np.pi
        angLattice[3:6] = angles / 180.0 * np.pi

        # The following is constraint 1)
        # check whether none of the angles is two small. Note that the problem is
        # symmetric around 90 degrees, that's the reason for 180 - minAngle.
        if np.where(angles < _MIN_ANGLE)[0].shape[0] != 0 or \
                np.where(angles > (180.0 - _MIN_ANGLE))[0].shape[0] != 0:
            lat_OK = False

        # The following is constraint 2)
        # We get the distance by dividing the volume (found by det(lattice)) by the area of any two vectors
        vol = abs(np.linalg.det(lattice))

        # We don't need dist variable anymore since the corresponding MATLAB condition is commented:
        # %if ~isempty(find(dist<minVectorLength))
        '''
        dist = np.zeros(3)
        dist[0] = vol/(angLattice[0, 0] * angLattice[0, 1] * sin(angLattice[0, 5]))
        dist[1] = vol/(angLattice[0, 0] * angLattice[0, 2] * sin(angLattice[0, 4]))
        dist[2] = vol/(angLattice[0, 1] * angLattice[0, 2] * sin(angLattice[0, 3]))
        '''

        if np.where(angLattice[0:3] < self.minVectorLength(system))[0].shape[0] > 0:
            lat_OK = False

        if np.where(np.isreal(lattice) is False)[0].shape[0] > 0:
            lat_OK = False

        # The following is constraint 3)
        # check whether none of the angles between the vector defining the lattice and the diagonal
        # of the parallelogram formed by other 2 vectors defining the lattice is too small.

        if lat_OK:  # if it's not 0 it means there are no 0-length vectors
            a_bc = lattice[0, 0] * (lattice[1, 0] + lattice[2, 0]) + lattice[0, 1] * (lattice[1, 1] + lattice[2, 1]) + \
                   lattice[0, 2] * (lattice[1, 2] + lattice[2, 2])
            ab_c = lattice[2, 0] * (lattice[1, 0] + lattice[0, 0]) + lattice[2, 1] * (lattice[1, 1] + lattice[0, 1]) + \
                   lattice[2, 2] * (lattice[1, 2] + lattice[0, 2])
            b_ca = lattice[1, 0] * (lattice[0, 0] + lattice[2, 0]) + lattice[1, 1] * (lattice[0, 1] + lattice[2, 1]) + \
                   lattice[1, 2] * (lattice[0, 2] + lattice[2, 2])

            # |b+c|^2=|b|^2+|c|^2+2|b||c|cos(bc):
            anglesDiag = np.zeros(3)
            anglesDiag[0] = np.arccos(a_bc / (angLattice[0] * np.sqrt(
                angLattice[1] ** 2 + angLattice[2] ** 2 + 2. * angLattice[1] * angLattice[2] * np.cos(angLattice[3]))))
            anglesDiag[1] = np.arccos(b_ca / (angLattice[1] * np.sqrt(
                angLattice[0] ** 2 + angLattice[2] ** 2 + 2. * angLattice[0] * angLattice[2] * np.cos(angLattice[4]))))
            anglesDiag[2] = np.arccos(ab_c / (angLattice[2] * np.sqrt(
                angLattice[1] ** 2 + angLattice[0] ** 2 + 2. * angLattice[1] * angLattice[0] * np.cos(angLattice[5]))))
            anglesDiag = anglesDiag * 180. / np.pi

            if np.where(anglesDiag < _MIN_DIAG_ANGLE)[0].shape[0] != 0 or \
                    np.where(anglesDiag > (180.0 - _MIN_DIAG_ANGLE))[0].shape[0] != 0:
                lat_OK = False

        return lat_OK

    def isGoodSystem(self, system: AtomicStructure) -> bool:
        """
        Check if the given system belongs to this configuration space.
        Includes :meth:`isGoodDistances` and
        :meth:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure.isMoleculesDistinct`.

        :type system: :class:`AtomicStructure`
        :param system: some system to be checked against its conformance to this space.
        :rtype: bool
        :return: True if system belongs to this configuration space, False otherwise.
        """
        isGoodSystem = self.isGoodDistances(system)
        if isGoodSystem and self.moleculesDistinctCheck:
            isGoodSystem = system.isMoleculesDistinct()
        return isGoodSystem

    def calcVolume(self):
        """
        This function estimates the volume occupied by a set of atoms described by chemical formula.

        :rtype: list
        :return: list of corresponding volumes.
        """
        volume = []
        for symbol in self.symbols:
            if symbol not in self.molecules:
                volume.append(calcVolume(self.externalPressure, symbol, self.volumeType))
            else:
                volume.append(sum(calcVolume(self.externalPressure, s, self.volumeType)
                                  for s in self.molecules[symbol]['symbols']))
        return np.array(volume)

    @property
    def systemFactory(self):
        """
        A reference to the class representing the system in this configuration space.

        :rtype: None
        :return: system class.
        """
        return None


class AtomisticConfig(ChemicalConfig):
    """
    Class with (more or less) most important parameters for the calculation that are set from the user input.
    """

    blocks = None
    fixed = None
    minAt = None
    maxAt = None
    isFixedComposition = None
    fingerprints = None

    def __init__(self, blocks: list=None, fixed: list=None, minAt: int=None, maxAt: int=None,
                 fingerprints: dict=None, sym_tolerance=None, **kwargs):
        """
        :type blocks: [[...],[...],...]
        :param blocks:
            each system of the configuration space must have composition
            which is span of rows of this parameter; obligatory
        :type fixed: [[...],[...],...]
        :param fixed:
            range for each block.
        :type minAt: int
        :param minAt:
            minimum number of atoms or molecules in the unit cell for the first generation.
        :type maxAt: int
        :param maxAt:
            maximum number of atoms or molecules in the unit cell for the first generation.
        :type fingerprints: dict
        :param fingerprints:
            arguments and keywords used to initialize :class:`~USPEX.Common.Fingerprints.Fingerprints.Fingerprints`.
        :type sym_tolerance: str or float
        :param sym_tolerance:
            a string ('high', 'medium' or 'low') or a number expressing symmetry tolerance.
        :type kwargs: dict
        :param kwargs:
            additional arguments and keywords used to initialize the parent class ChemicalConfig.
        """

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
        """
        Method which creates a dictionary representation of the config.

        :rtype: dict
        :return: dictionary representing the config.
        """
        dct = super().toDICT()
        dct['blocks'] = dct['blocks'].tolist()
        dct['fixed'] = dct['fixed'].tolist()
        return dct

    @classmethod
    def fromDICT(cls, dct: dict):
        """
        Method which reconstructs AtomisticConfig from its dictionary representation.

        :type dct: dict
        :param dct: dictionary representing the config.
        """
        dct['blocks'] = np.asarray(dct['blocks'])
        dct['fixed'] = np.asarray(dct['fixed'])
        return super().fromDICT(dct)

    def isGoodComposition(self, system: AtomicStructure) -> bool:
        """
        Method which checks if the structure meets the composition constraints.

        :type system: :class:`AtomicStructure`
        :param system: system to be checked.
        :rtype: bool
        :return: True if system meets the constraints, False otherwise.
        """
        if not set(system.composition.keys()) <= set(self.symbols):
            return False

        numIons = self.numIons(system.composition)
        numBlocks = self.numBlocks(system.composition)

        return np.all(np.dot(numBlocks, self.blocks) == numIons) and \
               np.all(numBlocks >= self.fixed[:,0]) and \
               np.all(numBlocks <= self.fixed[:,1]) and \
               self.minAt <= np.sum(numIons) <= self.maxAt

    def numIons(self, composition):
        """
        Creates numIons array from given composition.

        :type composition: dict
        :param composition: ('element' : 'amount')
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

    def numBlocks(self, composition) -> np.ndarray:
        """
        Creates numBlocks array from given composition.

        :type composition: dict
        :param composition: ('element' : 'amount')
        :rtype: list
        :return: list of blocks amounts corresponding *blocks* variable of this instance.
        """
        return np.round(np.linalg.lstsq(self.blocks.T, self.numIons(composition), rcond=None)[0]).astype(int)

    def randomComposition(self):
        """
        Creates random numIons array respecting configuration parameters: blocks, fixed, minAt and maxAt.

        :rtype: list
        :return: list of elements amounts corresponding *symbols* variable of this instance.
        """
        while True:
            numBlocks = np.fromiter((np.random.randint(low, high + 1) for low, high in self.fixed), dtype=int)
            numIons = np.dot(numBlocks, self.blocks)
            if self.minAt <= np.sum(numIons) <= self.maxAt:
                return numIons

    def isGoodSystem(self, system: AtomicStructure) -> bool:
        """
        Check if the given system belongs to this configuration space.
        Includes :meth:`isGoodDistances`,
        :meth:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure.isMoleculesDistinct` and
        :meth:`isGoodComposition`.

        :type system: :class:`AtomicStructure`
        :param system: Some system to be checked against its conformance to this space.
        :rtype: bool
        :return: True if system belongs to this configuration space, False otherwise.
        """
        return self.isGoodComposition(system) and super(AtomisticConfig, self).isGoodSystem(system)

    def findDesiredComposition(self, system1: AtomicStructure, system2: AtomicStructure, numIons_start: int,
                               debug: bool=False):
        """
        Find a composition that requires the least addition/deleting of atoms from child.

        :type system1: :class:`AtomicStructure`
        :param system1: first parent system.
        :type system2: :class:`AtomicStructure`
        :param system2: second parent system.
        :type numIons_start: int
        :param numIons_start: starting point for approximation.
        :type debug: bool
        :param debug: False by default. If set to True, use static values instead of random to reproduce results.
        :rtype: tuple of lists
        :return: (numIons, numBlocks) to determine which atoms could be used to make a child.
        """

        maxBlocks = self.numBlocks(system1.composition) + self.numBlocks(system2.composition)

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
                                self.fixed[blockOrder[i], 0] <= ind <= self.fixed[blockOrder[i], 1]:
                            min1 = np.sum(abs(composition_tmp - ind * block))
                            blockN[blockOrder[i]] = ind
                        ind += 1

                    composition_tmp -= blockN[blockOrder[i]] * block

                if bestGreed > np.sum(abs(numIons_start - np.dot(blockN, blocks))) and \
                                np.all(blockN >= self.fixed[:,0]) and \
                                np.all(blockN <= self.fixed[:,1]):
                    bestGreed = np.sum(abs(numIons_start - np.dot(blockN, blocks)))
                    numBlocks = blockN

            try:
                numIons = np.dot(numBlocks, blocks)
            except:
                numIons = None

        return numIons, numBlocks

    @property
    def systemFactory(self):
        """
        A reference to the class representing the system in this configuration space.

        :rtype: :class:`AtomicStructure`
        :return: system class.
        """
        return AtomicStructure
