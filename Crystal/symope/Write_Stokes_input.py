import numpy as np
from ase import Atoms


def Write_Stokes_input(numIons, minD, nsym, lat1, fixRndSeed, sym_coef):
    """
    The function prepares data for Stokes' program input.
    :param numIons: number of atoms.
    :param minD: minimum distances matrix.
    :param nsym: space group number.
    :param lat1: lattice.
    :param fixRndSeed: value of the fixed random seed.
    :param sym_coef: symmetry coefficient.
    :return content: content of the input file.
    """

    # Convert input values to NumPy arrays:
    #numIons = numIons_standardize(numIons)
    minD = np.asarray(minD, dtype=float)
    lat1 = np.asarray(lat1)

    # Delete 0 elements both from numIons and minDistance:
    to_delete = np.where(numIons == 0)

    numIons = np.delete(numIons, to_delete, axis=0)
    minD = np.delete(minD, to_delete, axis=0)
    minD = np.delete(minD, to_delete, axis=1)

    content = ''

    # Space group:
    content += '%4d ! space group \n' % nsym

    # Lattice:
    empty = Atoms(cell=lat1[0], pbc=True)
    content += '{:6.3f} ! volume of primitive unit cell\n'.format(empty.get_volume())
    #content += '%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f ! lattice of primitive unit cell\n' % tuple(lat1[0, 0:6])

    # Number of atoms part:
    content += '%4d ! number of types of atoms\n' % len(numIons)
    for i in range(len(numIons)):
        content += '%4d ' % numIons[i]
    content += '! number of atoms of each type\n'

    # Minimum distances part:
    if sym_coef is not None:
        minD = minD * sym_coef
    for ii in range(minD.shape[0]):
        for jj in range(minD.shape[1]):
            content += '%5.3f ' % minD[ii, jj]
    content += '! minimum distance between atoms \n'

    # Symmetrization coefficient part:
    if sym_coef is not None:
        content += ' 1 coefficient between minDist and symmetrization distance \n'

    # Random seeds part:
    if fixRndSeed > 0:
        content += '%10d %10d ! RandSeeds \n' % tuple([int(round(num * 10 ** 6)) for num in np.random.random(2)])

    return content
