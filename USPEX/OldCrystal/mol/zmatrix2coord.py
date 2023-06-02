"""
USPEX.Common.Atomistic.mol.zmatrix2coord
========================================

Function that transforms Z-matrix to XYZ coordinates

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np


def zmatrix2coord(zmatrix, fmt):
    """
    Function that transforms Z-matrix to XYZ coordinates. Remember that the Z-matrix of a molecule is defined
    in spherical coordinates, so we need a lot of transformations from (r, theta, phi) to (x, y, z).

    :type zmatrix: numpy array
    :param zmatrix:
        Z-matrix of a molecule.
    :type fmt: numpy array
    :param fmt:
        for each atom in the molecule are listed the indices of three other atoms,
        with respect to which the parameters of the Z-matrix are calculated.
    :rtype: numpy array
    :return:
        XYZ coordinates of atoms in the molecule.
    """
    N_atom = len(zmatrix)
    coords = np.zeros((N_atom, 3))
    origin = zmatrix[0, :]
    if N_atom > 1:
        coords[1, 2] = zmatrix[1, 0]*np.cos(zmatrix[1, 1])
        coords[1, 0] = zmatrix[1, 0]*np.sin(zmatrix[1, 1])*np.cos(zmatrix[1, 2])
        coords[1, 1] = zmatrix[1, 0]*np.sin(zmatrix[1, 1])*np.sin(zmatrix[1, 2])
        if N_atom > 2:
            for i in range(2, N_atom):
                if i == 2:
                    ref = coords[fmt[2, :2] - 1, :]
                else:
                    ref = coords[fmt[i, :] - 1, :]
                coords[i, :] = GetXYZ(ref, zmatrix[i, :])
    coords += origin
    return coords


def GetXYZ(ref, zmatrix):
    """
    Get the XYZ coordinates of the current atom from its Z-matrix coordinates
    and the XYZ coordinates of the reference atoms.

    :type ref: numpy array
    :param ref: XYZ coordinates of the reference atoms.
    :type zmatrix: numpy array
    :param zmatrix: Z-matrix coordinates of the current atom.
    :rtype: numpy array
    :return: XYZ coordinates of the current atom.
    """
    r = zmatrix[0]
    theta = zmatrix[1]
    phi = -zmatrix[2]

    coor = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
    u1 = ref[1, :] - ref[0, :]
    if len(ref) == 2:
        u2 = np.array([1, 0, 0])
    else:
        u2 = ref[2, :] - ref[1, :]

    z = u1/np.linalg.norm(u1)
    y = np.cross(u1, u2)
    y = y/np.linalg.norm(y)
    x = np.cross(y, z)
    x = x/np.linalg.norm(x)

    # coor = coor / (np.stack([x, y, z]).T)
    coor = np.linalg.lstsq(np.stack([x, y, z]), coor)[0]
    return coor + ref[0, :]
