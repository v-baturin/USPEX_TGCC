"""
USPEX.Common.Atomistic.Crystal
==============================

Class AtomicStructure-type with periodicity

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import spglib

from .AtomicStructure import AtomicStructure


class Crystal(AtomicStructure):
    """
    Class describing atoms-composed crystal structure with properties.
    Descendant of :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure`.

    Sets **pbc** for ase.Atoms component to [True, True, True].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, pbc=[True, True, True], **kwargs)

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
        return '{:7s} {:4s}'.format(*[str(x) for x in spglib.get_spacegroup(cell, symprec=1.0e-2).split()])
