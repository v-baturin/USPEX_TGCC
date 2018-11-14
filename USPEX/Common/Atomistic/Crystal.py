'''
@file        Crystal.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        16 November 2016
@brief       Class AtomicStructure-type + System-type structure with periodicity
'''

__author__ = 'p.bushlanov'

import spglib
import numpy as np

from .AtomicStructure import AtomicStructure

class Crystal(AtomicStructure):
    '''
    Class describing crystal Atoms-type structure with properties

    '''

    dimension = 3

    @property
    def symmetry(self):
        '''
        Property-method which calculates symmetry group for the structure.
        '''
        lattice = self.get_cell()
        coordinates = self.scaled_coordinates
        numbers = self.get_atomic_numbers()
        cell = (lattice, coordinates, numbers)
        return '{:7s} {:4s}'.format(*[str(x) for x in spglib.get_spacegroup(cell, symprec=1.0e-2).split()])
