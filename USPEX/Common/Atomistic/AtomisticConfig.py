'''
@file        Config.py
@author:     Pavel Bushlanov
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        5 December 2016
@brief       Class for description of configuration of target space.
'''


import numpy as np

from copy import copy

from ..Config import Config
from .AtomicStructure import AtomicStructure
from .Element import Element

# A minimal angle between any two vectors defining the lattice.
_MIN_ANGLE = 55

# A minimal angle between the vector defining the lattice and the
# diagonal of the parallelogram formed by other 2 vectors defining the lattice
_MIN_DIAG_ANGLE = 30


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

    def toDICT(self):
        dct = super().toDICT()
        dct['goodBonds'] = dct['goodBonds'].tolist()
        dct['valences'] = dct['valences'].tolist()
        dct['valenceElectrons'] = dct['valenceElectrons'].tolist()
        dct['minDistMatrice'] = dct['minDistMatrice'].tolist()
        dct['CenterminDistMatrice'] = dct['CenterminDistMatrice'].tolist()
        return dct

    @classmethod
    def fromDICT(cls, dct : dict):
        dct['goodBonds'] = np.asarray(dct['goodBonds'])
        dct['valences'] = np.asarray(dct['valences'])
        dct['valenceElectrons'] = np.asarray(dct['valenceElectrons'])
        dct['minDistMatrice'] = np.asarray(dct['minDistMatrice'])
        dct['CenterminDistMatrice'] = np.asarray(dct['CenterminDistMatrice'])
        return super().fromDICT(dct)

    # Here we check whether minVectorLength already specified in the input.
    # If not, we calculate it based on the chemical compound of current system (see. Manual)
    def minVectorLength(self, system : AtomicStructure) -> float:
        if self._minVectorLength is None:
            radii = np.fromiter((Element(symbol).covalent_radius for symbol in system.chemicalSymbols), dtype=float)
            return 1.8 * radii.max()
        else:
            return self._minVectorLength

    def isGoodDistances(self, system : AtomicStructure) -> bool:
        return system.isGoodDistances(self.chemicalSymbols, self.minDistMatrice)

    def isGoodCenterDistances(self, system : AtomicStructure, molSymbols) -> bool:
        if len(system) < 2:
            return True
        indices = np.fromiter((self.symbols.index(symbol) for symbol in molSymbols), dtype=int)
        cmDM = self.CenterminDistMatrice[np.meshgrid(indices, indices)]
        cmDM -= np.diag(np.diag(cmDM))
        return np.all(system.get_all_distances(mic=np.any(system.atoms.get_pbc())) >= cmDM.T)

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
        isGoodDistances = self.isGoodDistances(system)
        return isGoodDistances



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

    def toDICT(self):
        dct = super().toDICT()
        dct['blocks'] = dct['blocks'].tolist()
        dct['fixed'] = dct['fixed'].tolist()
        dct['magRatio'] = dct['magRatio'].tolist()
        return dct

    @classmethod
    def fromDICT(cls, dct : dict):
        dct['blocks'] = np.asarray(dct['blocks'])
        dct['fixed'] = np.asarray(dct['fixed'])
        dct['magRatio'] = np.asarray(dct['magRatio'])
        return super().fromDICT(dct)

    def isGoodComposition(self, system : AtomicStructure) -> bool:
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
        num = []
        for symbol in self.symbols:
            if symbol in composition:
                num.append(composition[symbol])
            else:
                num.append(0)
        return np.array(num, dtype=int)

    def numBlocks(self, composition) -> np.ndarray:
        return np.round(np.linalg.lstsq(self.blocks.T, self.numIons(composition))[0]).astype(int)

    def isGoodSystem(self, system : AtomicStructure) -> bool:
        return self.isGoodComposition(system) and super(AtomisticConfig, self).isGoodSystem(system)
