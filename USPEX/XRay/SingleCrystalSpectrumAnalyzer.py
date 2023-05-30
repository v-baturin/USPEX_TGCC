"""
USPEX.Common.XRay.SingleCrystalSpectrumAnalyzer
===============================================

Class which implements the fitness function for comparing single crystal X-ray spectra

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>
"""

import numpy as np

from pymatgen.core.structure import Structure

from .get_reflections import get_reflections


class SingleCrystalSpectrumAnalyzer(object):

    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, expReflections: list, cellParameters: tuple):
        """
        Initializes the class.

        :type expReflections: list
        :param expReflections: single crystal experimental reflections.
        :type cellParameters: tuple
        :param cellParameters: Cell parameters linked to the reflections.
        """
        self.exp_reflections = np.array(expReflections, dtype=object)
        self.cellParameters = cellParameters

    def analyze(self, system):
        """
        Special method which allows the '[]' operator. Calculates the agreement with the experimental X-ray spectrum.

        :type system: :class:`AtomicStructure` or one of its subclasses
        :param system: the crystal structure whose theoretical spectrum needs to be compared with experiment.
        :rtype: float
        :return: a fitness denoting how much the theoretical spectrum differs from the experiment.
        """
        structure = system.getAtomicStructure()
        elementList = list(structure.getComposition().keys())

        # cannot compute xraydistance if cell parameters differ from reference
        if not np.allclose(self.cellParameters, structure.getCell().getCellParameters()):
            system['singleCrystalSpectrumAnalyzer.xraydistance'] = np.inf

        # pure hydrogen gets low agreement
        elif len(elementList) == 1 and elementList[0].short_name == 'H':
            system['singleCrystalSpectrumAnalyzer.xraydistance'] = np.inf

        # compute xraydistance
        else:
            # create a pymatgen Structure object
            structure = Structure(lattice=structure.getCell().getCellVectors(),
                                  species=[el.short_name for el in structure.getAtomTypes()],
                                  coords=structure.getFractionalCoordinates())

            # compute minimum d spacing
            min_d_spacing = 10000
            for reflection in self.exp_reflections:
                hkl = list(reflection[1])
                d_spacing = structure.lattice.d_hkl(hkl)
                if d_spacing < min_d_spacing:
                    min_d_spacing = d_spacing

            min_d_spacing = np.floor(min_d_spacing * 1000) / 1000
            th_reflections = get_reflections(structure, min_d_spacing)
            th_reflections = np.array(th_reflections, dtype=object)

            # scale theoretical intensities according to experimental maximum
            th_reflections[:, 0] = th_reflections[:, 0] / max(th_reflections[:, 0]) * max(self.exp_reflections[:, 0])

            # compute R-factor
            numerator = 0
            denominator = 0
            for i_hkl, hkl, sigma_hkl in self.exp_reflections:
                th_hkls = [r[1] for r in th_reflections]

                if hkl in th_hkls:
                    index = th_hkls.index(hkl)
                elif (-hkl[0], -hkl[1], -hkl[2]) in th_hkls:
                    index = th_hkls.index((-hkl[0], -hkl[1], -hkl[2]))
                else:
                    raise ValueError(f'Experimental reflection {hkl} with intensity {i_hkl} not found in theory.')

                numerator += (1 / sigma_hkl ** 2) * (i_hkl - th_reflections[index][0]) ** 2
                denominator += (1 / sigma_hkl ** 2) * i_hkl ** 2

            wR = np.sqrt(numerator / denominator)       # weighted R-factor
            system.setProperty('singleCrystalSpectrumAnalyzer.xraydistance', wR)

    def xraydistance(self, system):
        if 'singleCrystalSpectrumAnalyzer.xraydistance' not in system:
            self.analyze(system)
        assert 'singleCrystalSpectrumAnalyzer.xraydistance' in system
        return system['singleCrystalSpectrumAnalyzer.xraydistance']

    @staticmethod
    def parse(hklFile: str):
        """
        It parses a file containing information about the experimental spectrum and
        it returns a dictionary containing the parameters for class initialization.

        :type filename: str
        :param filename: SHELXL hkl file containing the user-given reflections.
        :rtype: list
        :return: single crystal reflections parsed from the file.
        """
        expReflections = []
        with open(hklFile, 'r') as f:
            for line in f:
                values = line.split()

                # the hkl file terminates with all zeros
                if values == ['0', '0', '0', '0.00', '0.00']:
                    break

                i_hkl = float(values[3])
                if i_hkl > 0:
                    hkl = (int(values[0]), int(values[1]), int(values[2]))
                    sigma_hkl = float(values[4])
                    expReflections.append([i_hkl, hkl, sigma_hkl])
        return expReflections
