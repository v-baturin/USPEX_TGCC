"""
USPEX.Calculators.Common.KPoints
================================

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>

"""

import numpy as np


class BadKPoints(Exception):
    pass


class KPoints(object):
    def __init__(self, Kresol : float):
        assert isinstance(Kresol, float)
        self.Kresol = Kresol

    def build(self, cell):

        #angLattice = latConverter(system.lattice)
        cell = type(cell)(cell.getCellVectors(), (1,1,1))
        angLattice = cell.getCellParameters()

        dist = np.zeros(3)
        dist[2] = cell.getVolume() / (angLattice[0] * angLattice[1] * np.sin(angLattice[5]*np.pi/180))
        dist[1] = cell.getVolume() / (angLattice[0] * angLattice[2] * np.sin(angLattice[4]*np.pi/180))
        dist[0] = cell.getVolume() / (angLattice[1] * angLattice[2] * np.sin(angLattice[3]*np.pi/180))

        Kpoints = [int(x) for x in np.ceil(1.0 / (dist * self.Kresol))]

        # if abs(system.dimension) == 2:  # force Kpoints = 1 in z direction
        #     Kpoints[2] = 1

        if np.max(Kpoints) > 20 and cell.getVolume() > 50.0:  # to prevent some crazy lattices
            raise BadKPoints

        return Kpoints
