import numpy as np
from copy import copy


from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from ..Bonds import Bond, Bonds
from .MaxBonds import MaxBonds
from .AtomTypeCounter import atomTypeCounter

# import logging
# logger = logging.getLogger('SoftModeMutation')
#

def BondHardness(SYSTEM : AtomicStructure, goodBonds) -> Bonds:
    '''
    The function calculates bonds which make contribution to hardness.

    Meaning of some parameters (from empirical database (should be defined by user, otherwise use default)):
    - N_val - number of valence electrons;
    - val - valence.

    :param system: input system of class Crystal or Surface or Cluster
    :param goodBonds: good bonds (from USPEX defaults).
    :return bonds: list of Bonds which make contribution to hardness.
    '''

    '''
    Algorithm:
        1) Calculate all atomic pairs within a distance range [max_bond].
        2) Group all bonds within a distance range [same_bond].
        3) Consider all short bonds within a range [small_bond].
        4) If the atoms are not fully connected in 3D, add long bonds.
        5) Repeat 4 until 3D connectivity is satisfied.
    '''


    # The following parameters define how to evaluate the bond by distance:
    small_bond = -0.37 * np.log(np.array(goodBonds))  # maximum deviation to consider a bond

    N_atom = len(SYSTEM)
    atomTypes, atom_type_seq = atomTypeCounter(SYSTEM.chemicalSymbols)

    # 1) Calculate bonds within upper bound to max_bond.
    # 2) Group bonds by using same_bond criterion.
    bond_total = MaxBonds(SYSTEM)
    bond_group = np.unique([bond.type for bond in bond_total])
    bond_group_ref = np.copy(bond_group)

    # 3) Add bonds by group.
    bond_in = Bonds()

    nL = 2  # number of layers that we are considering to calculate hardness
    nL_ALL = 2*nL+1
    colors = np.zeros((2*nL+1, 2*nL+1, 2*nL+1, N_atom), dtype=int) # used to check connectivity - all colors are the same in this case
    Ncolors = ((2 * nL) ** 3) * N_atom
    for k1 in range(2 * nL + 1):
        for k2 in range(2 * nL + 1):
            for k3 in range(2 * nL + 1):
                for i in range(N_atom):
                    colors[k1, k2, k3, i] = (i+1) + ((2*nL+1) ** 2) * N_atom * k1 + (2*nL+1) * N_atom * k2 + N_atom * k3
    colors1 = np.copy(colors)    # Calculate bond valence using classical Brown's bond valence model.


    # 4) First to consider all the short bonds.
    for bondID in bond_group_ref:
        # bonds_tmp = bond_total.getType(bondID)
        # if len(bonds_tmp):
        for bond in bond_total.getType(bondID):
            origin, end = bond.atoms()
            dx, dy, dz = bond.direction
            a, b = atom_type_seq[origin], atom_type_seq[end]
            isBondUsed = False
            for m1 in range(2*nL+1):
                for m2 in range(2 * nL + 1):
                    for m3 in range(2 * nL + 1):
                        n1, n2, n3 = m1 + dx, m2 + dy, m3 + dz
                        if n1 < 0 or n2 < 0 or n3 < 0 or n1 > 2*nL or n2 > 2*nL or n3 > 2*nL:
                            continue
                        if colors[nL, nL, nL, end] == colors[nL+dx, nL+dy, nL+dz, origin] and bond.delta > small_bond[a,b]:
                            continue
                        isBondUsed = True
                        cj = colors1[m1, m2, m3, end]
                        ci = colors1[n1, n2, n3, origin]
                        for r1 in range(2*nL+1):
                            for r2 in range(2 * nL + 1):
                                for r3 in range(2 * nL + 1):
                                    for m in range(N_atom):
                                        if colors1[r1, r2, r3, m] == cj:
                                            colors1[r1, r2, r3, m] = ci
            if isBondUsed:
                bond.enable = True

        # next step is needed since we don't apply color periodic boundary conditions
        for i in range(N_atom):
            for j in range(i+1, N_atom):
                if colors1[nL, nL, nL, i] == colors1[nL, nL, nL, j]:
                    for m1 in range(2*nL+1):
                        for m2 in range(2*nL+1):
                            for m3 in range(2*nL+1):
                                if colors1[m1, m2, m3, i] != colors1[m1, m2, m3, j]:
                                    cj = colors1[m1, m2, m3, i]
                                    ci = colors1[m1, m2, m3, j]
                                    for r1 in range(2*nL+1):
                                        for r2 in range(2 * nL + 1):
                                            for r3 in range(2 * nL + 1):
                                                for m in range(N_atom):
                                                    if colors1[r1, r2, r3, m] == cj:
                                                        colors1[r1, r2, r3, m] = ci

        colors = np.copy(colors1)
        isConnected = True
        c1 = colors[0,0,0,0]
        c2 = None
        for k1 in range(nL-1, nL_ALL-1):
            for k2 in range(nL-1, nL_ALL-1):
                for k3 in range(nL-1, nL_ALL-1):
                    for i in range(N_atom):
                        if colors[k1,k2,k3,i] != c1 and not c2:
                            c2 = colors[k1,k2,k3,i]
                        elif colors[k1,k2,k3,i] != c1 and colors[k1,k2,k3,i] != c2:
                            isConnected = False

        if isConnected:
            break


    # if bonds_tmp[0].delta < small_bond[a, b]:
    #     bond_in += bonds_tmp
    #     bond_group = np.delete(bond_group, 0)

    # 5) Check 3D connectivity, if not satisfied, add more bonds, but we only include those bonds which could increase
    # connectivity, otherwise, we won't add them.

    for i, bond in enumerate(bond_total):
        if bond.enable:
            a, b = bond.atoms()
            if a == b:
                if i % 2 == 1:
                    bond_in += [bond]
            else:
                bond_in += [bond]

    # while len(List) < N_atom:
    #     if len(bond_group):  # it can come from MaxBonds()
    #         # logger.log('The structure is not fully connected, adding more bonds')
    #         bonds_tmp = bond_in + bond_total.getType(bond_group[0])
    #         List_new = bonds_tmp.connectList()
    #         bond_group = np.delete(bond_group, 0)
    #
    #         if len(List_new) > len(List):  # increase connectivity accept
    #             # logger.log('The connectivity is increased, accept adding more bonds')
    #             bond_in = copy(bonds_tmp)
    #             List = copy(List_new)
    #         else:
    #             pass
    #             # logger.log('The connectivity is not increased, reject adding more bonds')
    #     else:
    #         # logger.log('Running out of all bonds, has to exit')
    #         print('Running out of all bonds, has to exit')
    #         break

    return bond_in
'''
if __name__ == '__main__':
    from lib.Systems.Crystal.Crystal import Crystal
    scaled_positions = [[ 0., 0., 0.], [ 0.33333, 0.66667, 0.], [ 0., 0., 0.5], [ 0.66667, 0.33334, 0.5]]
    cell = [[ 2.456, 0., 0.],[-1.228, 2.126958, 0. ], [ 0., 0., 6.696]]
    system = Crystal(symbols=4 * ['C'], scaled_positions=scaled_positions, cell=cell)

    goodBonds = [[0.5]]

    bond_in_ref = []
    bond_in_ref.append(Bond(2, 3, -0.10204218, 1, [ -1. ,  -1. ,    0.]))
    bond_in_ref.append(Bond(0, 1, -0.10204198, 1, [  0. ,  -1. ,    0.]))
    bond_in_ref.append(Bond(0, 1, -0.10202091, 1, [  0. ,   0. ,    0.]))
    bond_in_ref.append(Bond(2, 3, -0.10202071, 1, [  0. ,   0. ,    0.]))
    bond_in_ref.append(Bond(2, 3, -0.10202071, 1, [ -1. ,   0. ,    0.]))
    bond_in_ref.append(Bond(0, 1, -0.10202071, 1, [ -1. ,  -1. ,    0.]))
    bond_in_ref.append(Bond(0, 2, 1.828 , 4, [  0. ,   0. ,   -1. ]))
    bond_in_ref.append(Bond(0, 2, 1.828 , 4, [  0. ,   0. ,    0. ]))

    atom_type_seq_ref = [0,0,0,0]
    R_val_ref = [0.76]

    bond_in = BondHardness(system, goodBonds)

    assert bond_in == bond_in_ref
'''
