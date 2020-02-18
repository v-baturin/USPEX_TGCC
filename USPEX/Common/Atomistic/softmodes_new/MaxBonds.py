from copy import copy

import numpy as np

from ase.neighborlist import primitive_neighbor_list
from scipy.spatial.distance import cdist

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from USPEX.Common.Atomistic.Element import Element
from ..Bonds import Bond, Bonds
from ..super_matrix import super_matrix
from .AtomTypeCounter import atomTypeCounter




def MaxBonds_new(system, cutoff:float=Bond.MAX_BOND) -> Bonds:

    symbols = system.chemicalSymbols

    bonds = Bonds()
    i_init, j_init, dists, vecs, dirs = primitive_neighbor_list(quantities='ijdDS', pbc=system.pbc,
                                                          cell=system.get_cell(complete=True),
                                                          positions=system.get_scaled_positions(),
                                                          cutoff=cutoff, numbers=system.numbers,
                                                          use_scaled_positions=True)

    for i, j, dist, vec, dir in zip(i_init, j_init, dists, vecs, dirs):
        # TODO Why we had this less 0.5A and not more than 5A (usually)
        # if np.abs(dist - tmp_Rval) > cutoff or dist < 0.5:
        if dist < 0.5:
            continue
        bonds.append(Bond(atom1=system[i], atom2=system[j], direction=dir, distance=dist, vector=vec))


    # bonds: (1) atom-i, (2) atom-j, (3) dist, (4) bond_type.
    # Now we assign the bond type

    # rank = np.argsort([b.distance for b in tmp_bonds])  # sort the bonds by dist from low to high
    # tmp_bonds = tmp_bonds[rank]
    tmp_bonds = sorted(bonds, key=lambda x: x.distance)

    # N_bonds = bonds.shape[0]
    bond_type:int = 0
    for bond in tmp_bonds:
        if not bond.type:
            bond_type += 1
            # symbols_ref = sorted(bond.symbols)
            # Obtain all bonds with the same type by distance:
            for b in [b for b in tmp_bonds if b == bond]:
                if not b.type:
                    b.type = bond_type
    return bonds


def MaxBonds(SYSTEM : AtomicStructure) -> Bonds:
    '''
    This program is to:
    - Obtain all bonds for a given structure.
    - Sort the bonds by distance.
    - Assign bond types.

    In general: 
    for each atom in system searched atoms in surrounding of Bond.MAX_BOND Angstrems (by default it's 5A) radius.
    For each atom in obtained list, if distance between them are bigger than 0.5A, then it's a bond.
    Type of the bond defined by distance.
    If 2 bonds has the same distance between atoms of the same types, then these bonds are of the same type.


    :param system: considered crystal, surface or cluster..
    :return: bonds: sorted bonds by distance
    '''

    Rmax = Bond.MAX_BOND

    N_atom = len(SYSTEM)
    atomType, atom_type_seq = atomTypeCounter(SYSTEM.chemicalSymbols)

    R_val = {}
    for type in atomType:
        R_val[type] = Element(type).covalent_radius

    '''
    ------    Search for all atomic pairs  --------------
    Here we count the coordination of each atom one by one.
    Firstly we construct the possible direction matrix such as [1 0 0; 0 1 0; 0 0 1; .......
    Ideally, the best way to is to do lattice optimization first, but we don't consider it for now.
    '''

    # --- This is the direction matrix.
    # --- maximally consider the direction of [1 2 0].

    directionMatrix = super_matrix(-1, 1, -1, 1, -1, 1)
    directionMatrix.remove([0, 0, 0])
    directionMatrix = np.array(directionMatrix, dtype=float)

    '''
    For each atom, we plot a sphere with Rmax as the radius and calculate the possible super cell size.
    1) Here we search for the optimum supercell for each atom. This way we don't need to make too big supercells.
       It can drastically reduce the cost when N is more then 100.
    2) Vectorize the part of distance matrix.
    '''

    bonds = np.zeros((0, 7))
    type1 = copy(atom_type_seq)
    coor1 = np.copy(SYSTEM.scaled_coordinates)
    N_atom1 = N_atom


    for i in range(N_atom):
        # Build the super cell matrix:
        # Here Target is scaled coordinates of atoms around cell with cut-off radius = Rmax.
        # Target = np.array([SYSTEM.scaled_coordinates[i] + Rmax / np.linalg.norm(np.dot(direction, SYSTEM.cell)) * direction for direction in directionMatrix])

        # lenX1, lenX2 = int(np.floor(min(Target[:, 0]))), int(np.floor(max(Target[:, 0])))
        # lenY1, lenY2 = int(np.floor(min(Target[:, 1]))), int(np.floor(max(Target[:, 1])))
        # lenZ1, lenZ2 = int(np.floor(min(Target[:, 2]))), int(np.floor(max(Target[:, 2])))

        lenX1, lenX2 = -2, 2
        lenY1, lenY2 = -2, 2
        lenZ1, lenZ2 = -2, 2

        Matrix_tmp = np.array(super_matrix(lenX1, lenX2, lenY1, lenY2, lenZ1, lenZ2))

        # if system.dimension == 0:
        #     Matrix_tmp = np.array(super_matrix(0, 0, 0, 0, 0, 0))
        # elif system.dimension == 2:
        #     Matrix_tmp = np.array(super_matrix(lenX1, lenX2, lenY1, lenY2, 0, 0))
        # else:
        #     Matrix_tmp = np.array(super_matrix(lenX1, lenX2, lenY1, lenY2, lenZ1, lenZ2))

        # Obtain the distances by vectorization, pdist2 is used in Matlab:
        N_Matrix = Matrix_tmp.shape[0]

        tmp_type = N_Matrix * type1  # using type1
        tmp_Rval = [R_val[atomType[x]] + R_val[atomType[atom_type_seq[i]]] for x in tmp_type]
        tmp_ID = np.tile(range(i, N_atom), N_Matrix)

        S_coor = np.tile(coor1, (N_Matrix, 1))  # using coor1
        S_Matrix = np.reshape(np.tile(Matrix_tmp, (1, N_atom1)).T, (3, N_atom1 * N_Matrix), order='F').T

        tmp_dist = cdist(np.reshape(np.dot(SYSTEM.scaled_coordinates[i, :], SYSTEM.cell), (1, 3)), np.dot(S_coor + S_Matrix, SYSTEM.cell))[0]

        To_Delete = np.where((np.abs(tmp_dist - tmp_Rval) > Rmax) | (tmp_dist < 0.5))[0]

        tmp_dist -= tmp_Rval

        N_total = N_Matrix * N_atom1  # total size of tmp_dist: N_Matrix*

        tmp2 = np.hstack((
            i * np.ones((N_total, 1)),
            np.copy(tmp_ID).reshape((N_total, 1)),
            np.copy(tmp_dist).reshape((N_total, 1)),
            np.zeros((N_total, 1)),
            np.copy(S_Matrix),
        ))

        tmp2 = np.delete(tmp2, To_Delete, axis=0)

        bonds = np.vstack((
            bonds,
            tmp2,
        ))

        # Discard the reference atom from now, to avoid double count of [i,j] pair:
        coor1 = np.delete(coor1, 0, axis=0)
        del type1[0]
        # type1 = np.delete(type1, 0, axis=0)
        N_atom1 -= 1

    # bonds: (1) atom-i, (2) atom-j, (3) dist, (4) bond_type.
    # Now we assign the bond type

    rank = bonds[:, 2].argsort()  # sort the bonds by dist from low to high
    bonds = bonds[rank, :]

    # N_bonds = bonds.shape[0]
    bond_type = 0
    bonds1 = Bonds()
    for bond in bonds:
        if bond[3] == 0:
            bond_type += 1
            # Obtain all bonds with the same type by distance:
            ID = np.where(bonds[:, 2] < bond[2] + Bond.SAME_BOND_THRESHOLD)[0]

            for j in range(len(ID)):
                bond1 = bonds[ID[j]]
                res1 = [atom_type_seq[int(x)] for x in bond[:2]]
                res2 = [atom_type_seq[int(x)] for x in bond1[:2]]
                if bond1[3] == 0 and res1 == res2:
                    bond1[3] = bond_type
                    bonds1.append(Bond(int(bond1[0]), int(bond1[1]), distance=bond1[2], type=bond_type, direction=[int(x) for x in bond1[4:]]))


    # for i in range(N_bonds):
    #     if bonds[i, 3] == 0:
    #         bond_type += 1
    #         # Obtain all bonds with the same type by distance:
    #         ID = np.where(bonds[:, 2] < bonds[i, 2] + Bond.SAME_BOND_THRESHOLD)[0]
    #
    #         for j in range(len(ID)):
    #             res1 = [atom_type_seq[int(x)] for x in bonds[i, :2]]
    #             res2 = [atom_type_seq[int(x)] for x in bonds[ID[j], :2]]
    #             if bonds[ID[j], 3] == 0 and res1 == res2:
    #                 bonds[ID[j], 3] = bond_type
    #                 bonds1.append(Bond(bonds[ID[j], 0], bonds[ID[j], 1], distance=bonds[ID[j], 2], type=bond_type, direction=bonds[ID[j], 4:]))


    return bonds1
