import numpy as np
from itertools import chain
from scipy.sparse.csgraph import connected_components

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from ..Bonds import connectList
from .MaxBonds import MaxBonds_new


_BONDS_CUTOFF = 5.0     # Angstroms

# def Bonds(system, cutoff=_BONDS_CUTOFF) -> Bonds:

def _connectedComponents(N, bonds):
    graph = np.zeros((27*N,27*N))
    for bond in chain(*bonds):
        i,j = bond.indicies
        k0,l0,m0 = bond.direction
        for k in range(3):
            for l in range(3):
                for m in range(3):
                    i_super = i + (9*k+3*l+m)*N
                    j_super = j + (9*(k+k0)+3*(l+l0)+(m+m0))*N
                    if 0 <= j_super < 27*N:
                        graph[i_super, j_super] = 1
    N_components, labels = connected_components(graph)
    return len(np.unique(labels[13*N:14*N]))


def BondHardness_new(SYSTEM : AtomicStructure, goodBonds) -> list:
    '''
    The function calculates bonds which make contribution to hardness.
    Used only for softmodemutation case and does not used for any other cases.

    NOTE (for future generations):
    this function should be used for softmodes calculation and for hardness calculation as well.
    But it is not right now. So, this method is not used for hardness calculation.
    Method that is above right now works proper for hardness calculation.

    :param SYSTEM:
    :param goodBonds:
    :return:
    '''

    N_atom = len(SYSTEM)


    # 1) Calculate bonds within upper bound to max_bond.
    # 2) Group bonds by using same_bond criterion.
    bond_total = MaxBonds_new(SYSTEM)

    # 3) Add bonds by group.
    bond_in = []
    bond_left = []

    # delete short bonds
    for bonds in bond_total:
        a,b = bonds[0].symbols
        small_bond = -0.37 * np.log(goodBonds[f'{a}-{b}'])
        if min([bond.delta for bond in bonds]) < small_bond:
            bond_in.append(bonds)    # Add by group
        else:
            bond_left.append(bonds)
    # del bond_group[0]

    # 5, check 3D connectivity, if not satisfied, add more bonds
    #   but we only include those bonds which could increase connectivity
    # ---Looks like we have to include all bonds before the connectivity changes
    #   otherwise, we won't add them

    N_components = _connectedComponents(N_atom, bond_in)
    # List = connectList(chain(*bond_in))

    while N_components > 1:
        # disp('The stuture is not fully connected, adding more bonds');
        bond_tmp = bond_in + [bond_left.pop(0)]
        # List_new = connectList(chain(*bond_tmp))
        N_components_new = _connectedComponents(N_atom, bond_tmp)
        # if len(List_new) > len(List) or len(List) == 1: # increase connectivity accept
        if N_components_new < N_components:
            # disp('The connectivity is increased, accept adding more bonds');
            # List = List_new
            N_components = N_components_new
            bond_in = bond_tmp
            # else
            # disp('The connectivity is not increased, reject adding more bonds');

    # 6, Remove double count of bond like [i,i] pair;
    for i, bonds_tmp in enumerate(bond_in):
        indicies = []
        for j, bond in enumerate(bonds_tmp):
            a,b = bond.indicies
            if a == b:
                indicies.append(j)
        for j in sorted(indicies[::2], reverse=True):
            del bond_in[i][j]

    return bond_in

