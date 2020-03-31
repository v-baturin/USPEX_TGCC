__author__ = 'etikhonov'

import numpy as np


def coord2Zmatrix(coords, Format):
    coords = np.copy(coords)
    Format = np.copy(Format)
    coords  = np.real(coords)

    Zmatrix = np.copy(coords)  #1st atom always = coords
    N_atom = coords.shape[0]

    if N_atom > 1:
        coords -= coords[0,:]
        # 2nd atom, define it in spherical coordinates systems
        Zmatrix[1,0] = np.real(np.linalg.norm(coords[1,:]))
        if coords[1,2] == 0:
            Zmatrix[1,1] = np.pi * 0.5
        else:
            Zmatrix[1,1] = np.arccos(coords[1,2]/Zmatrix[1,0])

        if coords[1,1] == 0:
            Zmatrix[1,2] = 0
        else:
            Zmatrix[1,2] = np.arctan2(coords[1,1],coords[1,0])

        for ind in range(2,N_atom):
            a1 = coords[ind,:]
            a2 = coords[Format[ind, 0] - 1, :]  # there and below: python indexing from 0
            a3 = coords[Format[ind, 1] - 1, :]
            Zmatrix[ind,0] = np.real(np.linalg.norm(a2-a1))
            Zmatrix[ind,1] = GetAngle(a1,a2,a3)
            if ind == 2: # the dihedral angle between 1-2-3 and XY plane
                a4 = a3 + np.array([1.0, 0.0, 0.0])
                #Zmatrix(ind,3) = -1*GetDihedral(a1,a2,a3,a4);
            else:
                a4 = coords[Format[ind, 2] - 1, :]
                #Zmatrix(ind,3) = -1*GetDihedral(a1,a2,a3,a4);

            Zmatrix[ind,2] = GetDihedral(a1,a2,a3,a4)

    Zmatrix = np.real(Zmatrix)
    return Zmatrix

#----------Bond Angle
def GetAngle(a1,a2,a3):
    v1 = a1 - a2
    v2 = a3 - a2
    angle = np.arccos(np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
    return angle

#-----------Torsion
def GetDihedral(a1,a2,a3,a4):
    p = a2 - a1
    q = a3 - a2
    r = a4 - a3
    n1 = np.cross(p,q)
    n2 = np.cross(q,r)
    torsion = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    center = (a1 + a2 + a3) / 3.0
    if np.dot(n1, a4 - center) < 0:
        torsion *=-1

    return torsion
