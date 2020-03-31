__author__ = 'mrakitin'

import numpy as np
from scipy.special import erf

from .local_order import local_order


def fingerprint(V, dist_matrix, structure, Rmax=10.0, sigma=0.03, delta=0.08):
    """
    Fingerprint calculation function.
    Reference: A.R. Oganov, M. Valle. How to quantify energy landscapes. J. Chem. Phys, 104504, 2009.

    :param V: volume of the cell.
    :param dist_matrix: distance matrix of size N*4 ([atomID, type1, type2, distance]), contains all the distances
    (<Rmax) from the given atom in the unit cell
    :param numIons: number of atoms (used for introducing the weight).
    :return order: order.
    :return fing: fingerprint.
    :return atom_fing: atomic fingerprint.
    """

    V = float(V)
    numIons = np.unique(structure.get_chemical_symbols(), return_counts=True)[1]
    N_type = numIons.shape[0]
    N_atom = np.sum(numIons)
    N_pair = dist_matrix.shape[0]  # the number of atomic pairs being considered
    normalizer = 1

    N_Bins = int(round(Rmax / float(delta)))
    fing = np.zeros((N_type ** 2, N_Bins))
    atom_fing = np.zeros((N_atom, N_type, N_Bins))
    sigma /= (2.0 * np.log(2.0)) ** 0.5
    sqrt2_sigm = sigma * 2.0 ** 0.5

    # Vectorization:
    N_bins = int(np.ceil(8 * sigma / delta))  # how many bins each dist can contribute

    # The following variables are vectors:
    interval = np.zeros((N_pair, 2))
    atom1 = dist_matrix[:, 0]
    type1 = dist_matrix[:, 1]
    type2 = dist_matrix[:, 2]
    btype1 = (type1 + 1.0 - 1.0) * N_type + (type2 + 1.0)
    R0 = dist_matrix[:, 3]
    R02 = R0 ** 2.0
    min_bin = np.floor((-4.0 * sigma + R0) / float(delta) + 0.5) + 1.0

    for i in range(N_bins):
        # Obtain the interval vectors:
        interval[:, 0] = 5.0 * np.sign(delta * (min_bin - 1.0) - R0)
        ID = np.where(abs(delta * (min_bin - 1.0) - R0) <= 5.0 * sqrt2_sigm)
        interval[ID[0], 0] = (delta * (min_bin[ID] - 1.0) - R0[ID]) / sqrt2_sigm

        interval[:, 1] = 5.0 * np.sign(delta * min_bin - R0)
        ID = np.where(abs(delta * min_bin - R0) <= 5.0 * sqrt2_sigm)
        interval[ID[0], 1] = (delta * min_bin[ID] - R0[ID]) / sqrt2_sigm
        '''
        for j in range(interval.shape[0]):
            row = '%12.8f %12.8f' % tuple(interval[j])
            print row
        '''

        my_erf = erf(interval)  # directly use erf function
        '''
        for j in range(my_erf.shape[0]):
            row = '%12.8f %12.8f' % tuple(my_erf[j])
            print row
        '''

        delt = 0.5 * (my_erf[:, 1] - my_erf[:, 0]) / R02  # used for fing
        '''
        for j in range(delt.shape[0]):
            row = '%12.8f' % delt[j]
            print row
        '''
        delt_type = delt / numIons[type2.astype(int)]  # used for atomfing
        '''
        for j in range(delt_type.shape[0]):
            row = '%12.8f' % delt_type[j]
            print row
        '''

        # Atomfing has 3 dimensions: we need to categorize by the following:
        for j in range(N_atom):
            tmp_ID1 = np.where(atom1 == j)  # 1st filter by atom ID
            delt_type1 = delt_type[tmp_ID1]
            '''
            for k in range(delt_type1.shape[0]):
                row = '%12.8f' % delt_type[k]
                print row
            '''
            min_bin1 = min_bin[tmp_ID1]
            '''
            for k in range(min_bin1.shape[0]):
                row = '%12.8f' % min_bin1[k]
                print row
            '''
            for k in range(N_type):
                tmp_ID2 = np.where(type2[tmp_ID1] == k)[0]  # 2nd filter by atom type
                '''
                for m in range(tmp_ID2.shape[0]):
                    row = '%12.8f' % tmp_ID2[m]
                    print row
                '''
                delt_type2 = delt_type1[tmp_ID2]
                '''
                for m in range(delt_type2.shape[0]):
                    row = '%12.8f' % delt_type2[m]
                    print row
                '''

                min_bin2 = min_bin1[tmp_ID2]
                '''
                for m in range(min_bin2.shape[0]):
                    row = '%12.8f' % min_bin2[m]
                    print row
                '''

                for m in range(N_Bins):
                    ID = np.where((min_bin2 - 1) == m)[0]  # 3rd filter by dist (bin)
                    atom_fing[j, k, m] += sum(delt_type2[ID])
        '''
        num = 27
        for m in range(atom_fing[:, :, num].shape[0]):
            row = '%12.8f %12.8f %12.8f' % tuple(atom_fing[m, :, num])
            print row
        '''

        # fing has 2 dimensions: we need to categorize by the following:
        for j in range(N_type * N_type):
            ID1 = np.where(btype1 - 1 == j)  # 1st filter by bond type
            delt1 = delt[ID1]
            '''
            for k in range(delt1.shape[0]):
                row = '%12.8f' % delt1[k]
                print row
            '''

            for k in range(N_Bins):
                ID = np.where(min_bin[ID1] - 1 == k)[0]  # 2nd filter by dist (bin)
                fing[j, k] += sum(delt1[ID])

        '''
        print
        for k in range(fing.shape[1]):
            row = '%12.8f ' * fing.shape[0] % tuple(np.transpose(fing)[k])
            print row
        '''

        min_bin += 1  # move to the next neighboring bin

    atom_fing = atom_fing * V / (4.0 * np.pi * delta) - normalizer
    '''
    num = 124
    for m in range(atom_fing[:, :, num].shape[0]):
        row = '%12.8f %12.8f %12.8f' % tuple(atom_fing[m, :, num])
        print row
    '''

    for i in range(N_type):
        for j in range(N_type):
            fing[i * N_type + j, :] = fing[i * N_type + j, :] * V / (
                4.0 * np.pi * numIons[i] * numIons[j] * delta) - normalizer

    order = local_order(V, numIons, atom_fing, delta)

    return order, fing, atom_fing


if __name__ == "__main__":
    from read_poscar import read_poscar

    lat, atomType, numIons, coor = read_poscar('POSCAR_8')

    from make_matrices import make_matrices

    V, dist_matrix = make_matrices(lat, coor, numIons)

    order, fing, atom_fing = fingerprint(V, dist_matrix, numIons)

    for k in range(fing.shape[1]):
        print('%12.8f ' * fing.shape[0] % tuple(np.transpose(fing)[k]))
