'''
@file        KPoints.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        October 2016
@brief       Abstract class for all engines in USPEX code
'''


import numpy as np


class BadKPoints(Exception):
    pass


class KPoints(object):
    def __init__(self, Kresol : float):
        assert isinstance(Kresol, float)
        self.Kresol = Kresol

    def build(self, system):

        #angLattice = latConverter(system.lattice)
        angLattice = system.get_cell_lengths_and_angles()
        vol = abs(np.linalg.det(system.get_cell()))

        dist = np.zeros(3)
        dist[2] = system.volume / (angLattice[0] * angLattice[1] * np.sin(angLattice[5]*np.pi/180))
        dist[1] = system.volume / (angLattice[0] * angLattice[2] * np.sin(angLattice[4]*np.pi/180))
        dist[0] = system.volume / (angLattice[1] * angLattice[2] * np.sin(angLattice[3]*np.pi/180))

        Kpoints = [int(x) for x in np.ceil(1.0 / (dist * self.Kresol))]

        # if abs(system.dimension) == 2:  # force Kpoints = 1 in z direction
        #     Kpoints[2] = 1

        if np.max(Kpoints) > 20 and system.volume > 50.0:  # to prevent some crazy lattices
            raise BadKPoints

        return Kpoints
