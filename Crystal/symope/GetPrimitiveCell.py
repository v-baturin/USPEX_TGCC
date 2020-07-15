import numpy as np

from .latConverter import latConverter


def GetPrimitiveCell(sGroup, lat1, numIons, fixLat):
    """
    The function converts a conventional cell to a primitive one. Stokes' code deals with the primitive cell.
    For the prediction of a non-primitive cell, we need to get the primitive cell for the input.
    :param sGroup: space group symbol.
    :param lat1: lattice (1x6, angles in degrees).
    :param numIons: list with number of atoms.
    :param fixLat: a flag indicating if the lattice is fixed or not.
    :return numIons: number of atoms for the primitive cell.
    :return lat1: lattice for the primitive cell.
    :return Error: error flag.
    """

    Error = 0

    if sGroup[0] != 'P' and fixLat == 1:  # non-primitive cell
        numIons0 = np.copy(numIons)
        lat0 = np.copy(lat1)

        if sGroup[0] == 'F':
            Duplicate = 4

        elif sGroup[0] == 'R':
            doedl = np.copy(lat1[0, 3:6])
            doedl.sort()

            if np.linalg.norm(doedl - [90.0, 90.0, 120.0]) < 3.0:  # hexagonal unit cell
                Duplicate = 3

            else:
                # Rhombohedral unit cell, already primitive, but Stokes' code wants the form of hexagonal cell:
                Duplicate = 1
                lat2 = latConverter(lat1)

                # Make hexagonal from  primitive rhombohedral:
                lat2 = np.dot(np.asarray([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 1.0, 1.0]]), lat2)

                lat1 = latConverter(lat2)
                lat1[0, 3:6] *= (180.0 / np.pi)  # in degrees

        else:  # base centered cells (A,B,C)
            Duplicate = 2

        numIons = np.asarray(np.round(numIons / float(Duplicate))).astype(int)

        lat1[0, 0:3] /= (Duplicate ** (1.0 / 3.0))

        if np.sum(numIons0) % np.sum(numIons) != 0:  # nearly not needed
            print('Impossible to place %s atoms for %s symmetry.' % (str(numIons[0]), sGroup))
            Error = 1

    return numIons, lat1, Error


if __name__ == "__main__":
    sGroup = 'C2m'  # 12
    lat1 = np.asarray([[12.17, 2.99, 4.83, 90, 98.25, 90]])
    numIons = np.asarray([[4, 10]])
    fixLat = 1

    numIons, lat1, Error = GetPrimitiveCell(sGroup, lat1, numIons, fixLat)
    print('')
