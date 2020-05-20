"""
USPEX.Common.Atomistic.Crystal
==============================

Class AtomicStructure-type with periodicity

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import spglib
import numpy as np
from copy import copy

from .AtomicStructure import AtomicStructure
from .Element import Element
from ..XRay.SpectrumAnalyzer import SpectrumAnalyzer


# A minimal angle between any two vectors defining the lattice.
_MIN_ANGLE = 55

# A minimal angle between the vector defining the lattice and the
# diagonal of the parallelogram formed by other 2 vectors defining the lattice
_MIN_DIAG_ANGLE = 30


# Default symmetry tolerance
_DEFAULT_SYMMETRY_TOLERANCE = 0.05

class Crystal(AtomicStructure):
    """
    Class describing atoms-composed crystal structure with properties.
    Descendant of :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure`.

    Sets **pbc** for ase.Atoms component to [True, True, True].
    """

    def __init__(self, *args, minVectorLength: float = None, xraydata=None, sym_tolerance=None, **kwargs):
        """
        Initializes the class.

        :param args:
        :type minVectorLength: float
        :param minVectorLength:
            sets the minimum length of a cell parameter of a newly generated structure.
        :param xraydata:
        :type sym_tolerance: str or float
        :param sym_tolerance:
            a string ('high', 'medium' or 'low') or a number expressing symmetry tolerance.
        :type kwargs: dict
        :param kwargs: additional arguments and keywords used to initialize the parent class AtomisticConfig.
        """
        if 'pbc' in kwargs:
            assert np.allclose(np.asarray(kwargs['pbc'], dtype=int), np.array([1, 1, 1]))
            del kwargs['pbc']
        super().__init__(*args, pbc=[True, True, True], **kwargs)
        self.crystalConfig = {}
        if minVectorLength is not None:
            self.crystalConfig['minVectorLength'] = minVectorLength
        if xraydata is not None:
            assert isinstance(xraydata, dict)
            self.crystalConfig['xraydata'] = xraydata
        self._spectrumAnalyzer = None

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


    @property
    def symmetry(self):
        """
        Calculate symmetry group for the structure.

        :rtype: str
        :return: symmetry group symbol and number.
        """
        lattice = self.get_cell()
        coordinates = self.scaled_coordinates
        numbers = self.get_atomic_numbers()
        cell = (lattice, coordinates, numbers)
        return '{:7s} {:4s}'.format(*[str(x) for x in spglib.get_spacegroup(cell, symprec=self.sym_tolerance).split()])

    @property
    def xraydistance(self):
        """
        Method which takes a structure and calculates the distance (fitness function)
        between calculated and experimental X-ray spectrum.

        :rtype: float
        :return: distance between calculated and experimental spectrum.
        """
        if self._spectrumAnalyzer is None:
            if 'xraydata' in self.crystalConfig:
                self._spectrumAnalyzer = SpectrumAnalyzer(**self.crystalConfig['xraydata'])
            else:
                raise RuntimeError('Cannot optimize the quantity xraydistance. No experimental X-ray data found.')
        return self._spectrumAnalyzer(self)


    @property
    def minVectorLength(self):
        """
        :rtype: float
        :return: minimal vector length.
        """
        uniqueSimbols = np.unique(self.chemicalSymbols)
        radii = np.fromiter((Element(symbol).covalent_radius for symbol in uniqueSimbols), dtype=float)
        minVectorLength = self.crystalConfig['minVectorLength'] if 'minVectorLength' in self.crystalConfig else 1.8 * np.max(radii)
        assert isinstance(minVectorLength, float) and minVectorLength > 0
        return minVectorLength

    def isGoodLattice(self) -> bool:
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

        lattice = self.cell
        angLattice = self.get_cell_lengths_and_angles()
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

        if np.where(angLattice[0:3] < self.minVectorLength)[0].shape[0] > 0:
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

    def isGoodSystem(self):
        return super().isGoodSystem() and self.isGoodLattice()

    def toDICT(self, old: bool = True) -> dict:
        dct = super().toDICT(old)
        if not old and self.crystalConfig:
            dct['configuration']['crystal'] = self.crystalConfig
        return dct

    @classmethod
    def fromDICT(cls, dct: dict, old: bool = True):
        structure = super().fromDICT(dct, bool)
        if not old and 'configuration' in dct and 'crystal' in dct['configuration']:
            structure.crystalConfig = dct['configuration']['crystal']
        return structure