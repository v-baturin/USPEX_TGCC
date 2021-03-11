"""
    Fingerprint calculation function.
    Reference: A.R. Oganov, M. Valle. How to quantify energy landscapes. J. Chem. Phys, 104504, 2009.
"""

import numpy as np
from typing import Dict, Tuple, Union
from collections.abc import Mapping
from scipy.special import erf
from scipy.spatial.distance import cdist
from itertools import combinations


RMAX_DEFAULT = 10.0
SIGMA_DEFAULT = 0.03
DELTA_DEFAULT = 0.08

TOLERANCE_DEFAULT = 0.008


class Fingerprint(Mapping):
    def __init__(self, value : dict, weights):
        sizes = [len(v) for v in value.values()]
        assert len(sizes) > 0
        self._value = value
        self._weights = weights
        self._size = sizes[0]
        super().__init__()

    @property
    def value(self):
        return self._value

    @property
    def size(self):
        return self._size

    @property
    def weights(self):
        return self._weights

    @property
    def order(self):
        # eq. 5 in CPC-2010
        return np.sqrt(np.sum(np.fromiter((self._weights[key] * np.sum(self._value[key] ** 2)
                                           for key in self._value.keys()), dtype=float)))

    def __repr__(self):
        return self._value.__repr__()

    def __len__(self):
        return self._value.__len__()

    def __iter__(self):
        return self._value.__iter__()

    def __getitem__(self, item):
        if item in self._value:
            return self._value.__getitem__(item)
        else:
            return -np.ones(self._size)

    @staticmethod
    def cosine_distance(fingerprint1, fingerprint2):
        """
        Calculation of cosine distances using eq.(6b) from JCP-2009.
        """
        value1 = fingerprint1.value
        weights1 = fingerprint1.weights
        value2 = fingerprint2.value
        weights2 = fingerprint2.weights
        coef1 = 0
        coef2 = 0
        coef3 = 0
        for key in set(value1.keys()) | set(value2.keys()):
            fing1 = value1[key] if key in value1 else np.zeros((1), dtype=float)
            fing2 = value2[key] if key in value2 else np.zeros((1), dtype=float)
            weight1 = weights1[key] if key in weights1 else 0
            weight2 = weights2[key] if key in weights2 else 0
            coef1 += np.sqrt(weight1 * weight2) * np.sum(fing1 * fing2)
            coef2 += weight1 * np.sum(fing1 * fing1)
            coef3 += weight2 * np.sum(fing2 * fing2)
        dist = (1 - coef1 / (coef2 * coef3) ** 0.5) / 2
        return dist



class RadialDistributionUtility(object):

    def __init__(self, Rmax=RMAX_DEFAULT, sigma=SIGMA_DEFAULT, delta=DELTA_DEFAULT, tolerance=TOLERANCE_DEFAULT):
        """
        :type Rmax: float
        :param Rmax: threshold distance between i-th anf j-th atom.
        :type sigma: float
        :param sigma: smearing parameter.
        :type delta: float
        :param delta: bin width.
        :type tolerance: float
        :param tolerance:
        """
        self.Rmax = Rmax
        self.sigma = sigma
        self.delta = delta
        self.tolerance = tolerance

    def structureFingerprint(self, system):
        if not 'radialDistribitionUtility.structureFingerprint' in system:
            self._calcFingerprint(system)
        return system['radialDistribitionUtility.structureFingerprint']

    def atomFingerprints(self, system):
        if not 'radialDistribitionUtility.atomFingerprints' in system:
            self._calcFingerprint(system)
        return system['radialDistribitionUtility.atomFingerprints']

    def order(self, system):
        if not 'radialDistribitionUtility.order' in system:
            molecules = system['molecules']
            systemFactory = type(molecules[1])
            structure, disassembler = systemFactory.assemble(**system)
            if len(structure) == 0:
                order =  np.nan
            else:
                order = np.fromiter((atomFing.order for atomFing in self.atomFingerprints(system)), dtype = float)
                order *= np.sqrt(self.delta / (structure.getCell().getVolume() / len(structure)) ** (1.0 / 3.0))
            system['radialDistribitionUtility.order'] = order
        return system['radialDistribitionUtility.order']

    def averageOrder(self, system):
        '''
        :rtype: float
        :return: average local order for the structure.
        '''
        order = np.asarray(self.order(system), dtype=float)
        if np.any(np.isfinite(order)):
            a_order = np.mean(order[np.isfinite(order)])
        else:
            a_order = np.nan
        return a_order

    def structureOrder(self, system):
        """
        Calculate structure order.

        """
        if not 'radialDistribitionUtility.structureOrder' in system:
            molecules = system['molecules']
            systemFactory = type(molecules[1])
            structure, disassembler = systemFactory.assemble(**system)
            fingerprint = self.structureFingerprint(system)
            s_order = fingerprint.order * np.sqrt(self.delta / (structure.getCell().getVolume / len(structure)) ** (1.0 / 3.0))
            system['radialDistribitionUtility.structureOrder'] = s_order
        return system['radialDistribitionUtility.structureOrder']

    def quasientropy(self, system):
        """
        Calculate structure quasientropy.

        :type structure: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
        :param structure: system for which we want to calculate quasientropy.
        :type atom_fing: numpy array
        :param atom_fing: atomic fingerprint.
        :rtype: numpy array
        :return: structure quasientropy.
        """

        if not 'radialDistribitionUtility.quasientropy' in system:
            molecules = system['molecules']
            systemFactory = type(molecules[1])
            structure, disassembler = systemFactory.assemble(**system)
            atomFing = self.atomFingerprints(system)
            uniqueSymbols, inverse, numIons = np.unique(structure.getAtomTypes(),
                                                        return_inverse=True,
                                                        return_counts=True)
            sQE = 0.0
            weight = numIons / np.sum(numIons)

            for i in range(numIons.shape[0]):
                if numIons[i] > 1:
                    tmp = 0
                    indices = np.flatnonzero(inverse == i)
                    comb = list(combinations(indices, 2))
                    for j1, j2 in comb:
                        tmp_fing1 = atomFing[j1]
                        tmp_fing2 = atomFing[j2]

                        dist = Fingerprint.cosine_distance(tmp_fing1, tmp_fing2)

                        '''
                        if abs(dist - 1.0) < 0.000001:
                            dist = 0.99999
                        '''

                        tmp += (1 - dist) * np.log(1 - dist)

                    sQE += weight[i] * tmp / len(comb)
            system['radialDistribitionUtility.quasientropy'] = -sQE
        return system['radialDistribitionUtility.quasientropy']

    def clean(self, system):
        if 'radialDistribitionUtility.structureFingerprint' in system:
            del system['radialDistribitionUtility.structureFingerprint']
        if 'radialDistribitionUtility.structureOrder' in system:
            del system['radialDistribitionUtility.structureOrder']
        if 'radialDistribitionUtility.atomFingerprints' in system:
            del system['radialDistribitionUtility.atomFingerprints']
        if 'radialDistribitionUtility.order' in system:
            del system['radialDistribitionUtility.order']
        if 'radialDistribitionUtility.quasientropy' in system:
            del system['radialDistribitionUtility.quasientropy']

    def _calcFingerprint(self, system):
        """
        Calculates fingerprint and related things.
        """
        molecules = system['molecules']
        systemFactory = type(molecules[1])
        structure, disassembler = systemFactory.assemble(**system)
        uniqueSimbols, inverse, numIons = np.unique(structure.getAtomTypes(), return_inverse=True, return_counts=True)
        indices = np.argsort(inverse)
        revertIndices = np.argsort(indices)
        coordinates = structure.getFractionalCoordinates()[indices]
        dist_matrix = make_matrices(coordinates, structure.getCell().getCellVectors(), numIons, Rmax=self.Rmax)

        V = structure.getCell().getVolume()
        N_type = numIons.shape[0]
        N_atom = np.sum(numIons)
        N_pair = dist_matrix.shape[0]  # the number of atomic pairs being considered
        normalizer = 1

        N_Bins = int(round(self.Rmax / float(self.delta)))
        fing = np.zeros((N_type ** 2, N_Bins))
        atom_fing = np.zeros((N_atom, N_type, N_Bins))
        sigma = self.sigma / (2.0 * np.log(2.0)) ** 0.5
        sqrt2_sigm = sigma * 2.0 ** 0.5

        # Vectorization:
        N_bins = int(np.ceil(8 * sigma / self.delta))  # how many bins each dist can contribute

        # The following variables are vectors:
        interval = np.zeros((N_pair, 2))
        atom1 = dist_matrix[:, 0]
        type1 = dist_matrix[:, 1]
        type2 = dist_matrix[:, 2]
        btype1 = (type1 + 1.0 - 1.0) * N_type + (type2 + 1.0)
        R0 = dist_matrix[:, 3]
        R02 = R0 ** 2.0
        min_bin = np.floor((-4.0 * sigma + R0) / float(self.delta) + 0.5) + 1.0

        for i in range(N_bins):
            # Obtain the interval vectors:
            interval[:, 0] = 5.0 * np.sign(self.delta * (min_bin - 1.0) - R0)
            ID = np.where(abs(self.delta * (min_bin - 1.0) - R0) <= 5.0 * sqrt2_sigm)
            interval[ID[0], 0] = (self.delta * (min_bin[ID] - 1.0) - R0[ID]) / sqrt2_sigm

            interval[:, 1] = 5.0 * np.sign(self.delta * min_bin - R0)
            ID = np.where(abs(self.delta * min_bin - R0) <= 5.0 * sqrt2_sigm)
            interval[ID[0], 1] = (self.delta * min_bin[ID] - R0[ID]) / sqrt2_sigm
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

        atom_fing = atom_fing * V / (4.0 * np.pi * self.delta) - normalizer
        '''
        num = 124
        for m in range(atom_fing[:, :, num].shape[0]):
            row = '%12.8f %12.8f %12.8f' % tuple(atom_fing[m, :, num])
            print row
        '''

        for i in range(N_type):
            for j in range(N_type):
                fing[i * N_type + j, :] = fing[i * N_type + j, :] * V / (
                        4.0 * np.pi * numIons[i] * numIons[j] * self.delta) - normalizer

        n = len(uniqueSimbols)
        fing = {(s1.short_name, s2.short_name): fing[i * n + j] for i, s1 in enumerate(uniqueSimbols) for j, s2 in enumerate(uniqueSimbols)}
        atomFing = []
        weights = {s.short_name: w for s, w in zip(uniqueSimbols, numIons / np.sum(numIons))}
        for i in revertIndices:
            f = Fingerprint(value={s.short_name: atom_fing[i, j] for j, s in enumerate(uniqueSimbols)},
                            weights=weights)
            atomFing.append(f)

        system['radialDistribitionUtility.atomFingerprints'] = atomFing
        system['radialDistribitionUtility.structureFingerprint'] = Fingerprint(value=fing,
                                                                               weights=self.fingerprintWeights(structure))


    def dist(self, system1, system2):
        return Fingerprint.cosine_distance(self.structureFingerprint(system1), self.structureFingerprint(system2))

    def equal(self, system1, system2):
        return self.dist(system1, system2) < self.tolerance

    @staticmethod
    def fingerprintWeights(structure):
        '''
        :rtype: Dict[Tuple[str,str], float]
        :return: weights of fingerprints of each atom type pair to be used in cosine distance calculation.
        '''
        comp = structure.getComposition()
        # TODO Whether we really need to duplicate weights Like Fe-C and C-Fe
        weights = {(s1.short_name, s2.short_name): am1*am2 for s1, am1 in comp.items() for s2, am2 in comp.items()}
        weightSum = np.sum(list(weights.values()))
        for key in weights:
            weights[key] /= weightSum
        return weights


def super_matrix(xmin: int, xmax: int, ymin: int, ymax: int, zmin: int, zmax: int):
    """
    This is a small utility to quickly generate a 3d matrix series of the type [x1 y1 z1; x2 y2 z2; ......]
    with x, y and z in the input ranges [x_min, x_max], [y_min, y_max] and [z_min, z_max].
    The functional is usually used to create supercells.
    Example:
        INPUT:  [0 1 0 1 0 1]
        OUTPUT: [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]

    :type xmin: int
    :param xmin: min x value.
    :type xmax: int
    :param xmax: max x value.
    :type ymin: int
    :param ymin: min y value.
    :type ymax: int
    :param ymax: max y value.
    :type zmin: int
    :param zmin: min z value.
    :type zmax: int
    :param zmax: max z value.
    :rtype: list
    :return matrix: resulted matrix.
    """

    # matrix = []
    # for i in range(xmin, xmax + 1):
    #     for j in range(ymin, ymax + 1):
    #         for k in range(zmin, zmax + 1):
    #             matrix.append([i, j, k])
    # return matrix
    return [[i, j, k] for i in range(xmin, xmax + 1) for j in range(ymin, ymax + 1) for k in range(zmin, zmax + 1)]


def make_matrices(coor: np.ndarray, lat: np.ndarray, numIons: np.ndarray, Rmax=10.0):
    """
    The function prepares matrices for fingerprint calculation.

    :type coor: numpy array
    :param coor: coordinates of atoms of system for which we want to calculate the distance matrix,
     groupped according atom types.
    :type lat: numpy array
    :param lat: cell of system for which we want to calculate the distance matrix.
    :type numIons: numpy array
    :param numIons: number atoms of each type in system for which we want to calculate the distance matrix.
    :type Rmax: float
    :param Rmax: distance cutoff.
    :rtype: numpy array
    :return: distance matrix of the form [atom_i, atom_j, atomi_type, atomj_type, dist].
    """

    assert isinstance(Rmax, float) and Rmax >= 0

    coor = coor - np.floor(coor)  # scale it the [0 1]
    N_atom = np.sum(numIons)

    types = []
    for i in range(len(numIons)):
        types += [i] * numIons[i]
    types = np.asarray(types)

    dist_matrix = np.zeros((0, 4), dtype=float)

    '''
    Search for all atomic pairs:
    Here we count the coordination of each atom one by one.
    Firstly we construct the possible direction matrix such as [1 0 0; 0 1 0; 0 0 1; ...
    Ideally, the best way to deal with the irregular shape is to do lattice optimization.
    But we don't consider it for now.
    '''

    # This is a direction matrix, maximally consider the direction of [1 2 0]:
    matrix = np.asarray(super_matrix(-2, 2, -2, 2, -2, 2))

    # The idea is to find all zero values [0, 0, 0] and to remove them to avoid division by zero later.
    to_delete = np.all(matrix == 0, axis=1)
    matrix = np.delete(matrix, np.where(to_delete), axis=0)

    for i in range(N_atom):
        target = np.zeros(matrix.shape)

        # Build the super cell matrix:
        for j in range(matrix.shape[0]):
            target[j] = Rmax / np.linalg.norm(np.dot(matrix[j, :], lat)) * matrix[j, :] + coor[i, :]

        lenX1 = int(np.floor(min(target[:, 0])))
        lenY1 = int(np.floor(min(target[:, 1])))
        lenZ1 = int(np.floor(min(target[:, 2])))
        lenX2 = int(np.floor(max(target[:, 0])))
        lenY2 = int(np.floor(max(target[:, 1])))
        lenZ2 = int(np.floor(max(target[:, 2])))

        matrix_tmp = np.asarray(super_matrix(lenX1, lenX2, lenY1, lenY2, lenZ1, lenZ2))

        # Obtain the distances by vectorization, pdist is used here:
        N_matrix = matrix_tmp.shape[0]
        S_coor = np.tile(coor, (N_matrix, 1))

        # S_Matrix  = reshape(repmat(Matrix_tmp, 1, N_atom)', 3, N_atom*N_Matrix)';

        S_matrix = np.tile(matrix_tmp, (1, N_atom))
        S_matrix = np.transpose(S_matrix)
        S_matrix = np.reshape(S_matrix, (3, N_atom * N_matrix), 'F')
        S_matrix = np.transpose(S_matrix)
        # For debug:
        # for j in range(S_matrix.shape[0]):
        #    print '%i ' * 3 % tuple(S_matrix[j])

        tmp_dist = cdist(np.reshape(np.dot(coor[i], lat), (1, 3)), np.dot(S_coor + S_matrix, lat))
        # For debug:
        # for j in range(tmp_dist.shape[1]):
        #    print '%.12f' % (float(tmp_dist[0][j]))

        tmp_type = np.tile(types, N_matrix)
        # For debug:
        # for j in range(tmp_type.shape[0]):
        #    print tmp_type[j]


        to_delete = np.where((tmp_dist > Rmax) | (tmp_dist < 0.5))
        # For debug:
        # for j in range(to_delete[1].shape[0]):
        #    print to_delete[1][j]

        tmp_dist = np.delete(tmp_dist, to_delete[1], axis=1)  # reset all the distances
        # For debug:
        # for j in range(tmp_dist.shape[1]):
        #    print '%.12f' % (float(tmp_dist[0][j]))

        tmp_type = np.delete(tmp_type, to_delete[1], axis=0)
        # For debug:
        # for j in range(tmp_type.shape[0]):
        #    print tmp_type[j]

        if tmp_dist.shape[1] > 0:  # sometimes you can meet an isolated atom
            tmp2 = np.zeros((tmp_dist.shape[1], 4))
            tmp2[:, 0] = i
            tmp2[:, 1] = types[i]
            tmp2[:, 2] = tmp_type
            tmp2[:, 3] = tmp_dist
            dist_matrix = np.vstack((dist_matrix, tmp2))

    return dist_matrix
