"""
USPEX.Common.Atomistic.latticeMutation
======================================

Contains function for lattice mutation

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np

from numpy import linalg as LA
from ase.geometry.cell import cell_to_cellpar


def latticeMutation(system, MUTATION_RATE=None):
    """
    Mutates the lattice of an input system and rerutns the corresponding strain matrix.

    :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure` or descendant
    :param system: the system whose lattice we want to mutate.
    :type MUTATION_RATE: float
    :param MUTATION_RATE: mutation rate.
    :rtype: numpy array
    :return: strain matrix associated to the mutation.
    """
    if MUTATION_RATE is None:
        MUTATION_RATE = 0.25

    # lattice mutation
    new_Lattice = np.zeros((3, 3))
    dummy = True
    while LA.det(new_Lattice) < 0.01 or dummy:
        dummy = False
        strainMatrix = np.zeros((3, 3))
        epsilons = np.random.rand(6) * MUTATION_RATE
        strainMatrix[0, 0] = 1 + epsilons[0]
        strainMatrix[1, 1] = 1 + epsilons[1]
        strainMatrix[2, 2] = 1 + epsilons[2]
        strainMatrix[0, 1] = epsilons[3]/2
        strainMatrix[1, 0] = epsilons[3]/2
        strainMatrix[0, 2] = epsilons[4]/2
        strainMatrix[2, 0] = epsilons[4]/2
        strainMatrix[1, 2] = epsilons[5]/2
        strainMatrix[2, 1] = epsilons[5]/2
        new_Lattice = np.dot(system.cell, strainMatrix)

    # scale the lattice to the volume we assume it approximately to be
    ratio = np.power(system.volume / LA.det(new_Lattice), 1/3)
    lattice = cell_to_cellpar(new_Lattice)
    lattice[:3] *= ratio
    system.set_cell(lattice)
    return strainMatrix
