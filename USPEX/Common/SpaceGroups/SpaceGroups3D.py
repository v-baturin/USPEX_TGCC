import os
import json
import numpy as np
from copy import copy, deepcopy
from collections import Sequence
from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from pymatgen.symmetry.groups import in_array_list


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
with open('{}/decompositions.json'.format(HOMEPATH), 'rt') as f:
    DECOMPOSITIONS = json.load(f)

TRANSLATIONS = np.array([[[1.0, 0.0, 0.0, 1.0],[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]],
                         [[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 1.0],[0.0, 0.0, 1.0, 0.0],[0.0, 0.0, 0.0, 1.0]],
                         [[1.0, 0.0, 0.0, 0.0],[0.0, 1.0, 0.0, 0.0],[0.0, 0.0, 1.0, 1.0],[0.0, 0.0, 0.0, 1.0]]])


def _generate_full_symmetry_ops(generators, supercell: tuple):
    """
    This function is modification of _generate_full_symmetry_ops method of SpaceGroup class of pymatgen.symmetry.groups
    module. Which is distributed under the terms of the MIT License. Copyright (c) Pymatgen Development Team.
    This function generates an envelope of the set of space group generators. It includes translations within supercell.
    :param generators:
    :param supercell:
    :return:
    """
    symm_ops = np.array(generators)
    for op in symm_ops:
        op[0:3, 3] = np.mod(op[0:3, 3], supercell)
    new_ops = symm_ops
    while len(new_ops) > 0:
        gen_ops = []
        for g in new_ops:
            temp_ops = np.einsum('ijk,kl', symm_ops, g)
            for op in temp_ops:
                op[0:3, 3] = np.mod(op[0:3, 3], supercell)
                ind = np.where(np.abs(np.asarray(supercell) - op[0:3, 3]) < 1e-4)
                op[ind, 3] = 0
                if not in_array_list(symm_ops, op):
                    gen_ops.append(op)
                    symm_ops = np.append(symm_ops, [op], axis=0)
        new_ops = gen_ops
    return symm_ops


def uniqueRows(arr, thresh=0.0, metric='euclidean'):
    """
    Returns subset of rows that are unique, in terms of Euclidean distance
    :param arr:
    :param thresh:
    :param metric:
    :return:
    """
    distances = squareform(pdist(arr, metric=metric))
    idxset = {tuple(np.nonzero(v)[0]) for v in distances <= thresh}
    return arr[[x[0] for x in idxset]]


class Group(object):
    """
    Class representing a space group.
    """

    def __init__(self, generators : list, dimensions : list):
        """
        Initialize Group object.
        :param generators:
        :param dimensions:
        :param translationsExtra:
        """
        self.generators = copy(generators)
        self.dimensions = copy(dimensions)
        self._operators = _generate_full_symmetry_ops(self.generators, (1,1,1))

    @property
    def operators(self):
        return self._operators

    def __contains__(self, op):
        """
        Check if this group contains provided operation.
        :param op: 4x4 float matrix representing operation.
        :return: True if the group contain operation, False otherwise.
        """
        return in_array_list(self.operators, op)

    def __call__(self, nodes):
        """
        Calculates orbits of provided positions with respect to this group.
        The orbit of a position is the set of positions which can be obtained by acting with operations of the group.
        :param nodes: array of positions.
        :return: list of orbits, each orbit is an array of positions.
        """
        return [uniqueRows(np.divmod(np.around(orbit, decimals=6), (1,1,1))[1], thresh=1.0e-4)
                for orbit in (np.dot(self.operators[:,:3,:3], nodes.T)
                              + self.operators[:,:3,3].reshape(self.operators.shape[0],3,1)).transpose((2, 0, 1))]

    def getAllSubgroups(self, supercell: tuple):
        """
        Enumerates all subgroups of this group which preserve given supercell.
        :param supercell: 3-tuple describing supercell.
        :return: Subgroups object.
        """
        generators = copy(self.generators)
        dimensions = copy(self.dimensions)

        extendedOperators = self.operators
        for m, translation in zip(supercell,TRANSLATIONS):
            if m > 1 and not in_array_list(extendedOperators, translation):
                generators.append(translation)
                dimensions.append(m)
                extendedOperators = _generate_full_symmetry_ops(generators, supercell)

        genNum = len(generators)
        inds = [] # Indices of generators preserving supercell.
        for i, generator in enumerate(generators):
            if np.allclose(np.abs(np.dot(generator[0:3,0:3], supercell)), supercell) \
                    and not np.allclose(generator, np.eye(4)):
                inds.append(i)

        generatorsCombinations = [((0,),tuple(range(genNum)))]
        for q in range(len(inds)):
            for subgroup in combinations(inds, q+1):
                remainder = tuple(set(inds) - set(subgroup))
                generatorsCombinations.append(((0,) + subgroup, (0,) + remainder))
        return Subgroups(generators, dimensions, generatorsCombinations, supercell)

    def getNotPositionInvariantSubgroups(self, position):
        """
        Enumerates all subgroups of this group which preserve given supercell and given position.
        :param position: 3-array describing atomic position.
        :param supercell: 3-tuple describing supercell.
        :return: Subgroups object.
        """
        allGenerators = [np.identity(4, dtype = np.float)]
        trivialGenerators = [0]
        nonTrivialGenerators = [0]
        for i, generator in reversed(list(enumerate(self.generators))):
            if not np.allclose(generator, np.identity(4, dtype = np.float)):
                envelope = Group(self.generators[:i], self.dimensions[:i])
                trivial = False
                for operation in envelope.operators:
                    modifiedGenerator = np.dot(operation, generator)
                    modifiedGenerator[0:3, 3] = np.mod(modifiedGenerator[0:3, 3], (1,1,1))
                    if np.allclose(np.mod(np.around(np.dot(modifiedGenerator[0:3,0:3], position)
                                          + modifiedGenerator[0:3,3], decimals=5), (1,1,1)), position, atol = 1e-4):
                        allGenerators.append(modifiedGenerator)
                        trivialGenerators.append(len(allGenerators) - 1)
                        trivial = True
                        break
                if not trivial:
                    allGenerators.append(generator)
                    nonTrivialGenerators.append(len(allGenerators) - 1)
        return Subgroups(allGenerators, self.dimensions, [(nonTrivialGenerators, trivialGenerators)], (1,1,1))

    @staticmethod
    def getWrapedGroup(generators : list, dimensions : list, supercell : tuple):
        """
        Creates wraped group. Supercell becomes cell.
        :return: Group object.
        """
        generators = deepcopy(generators)
        generators_final = []
        dimensions_final = []
        gens_span = []
        for gen, dim in zip(generators, dimensions):
            gen[0:3, 3] = np.divmod(gen[0:3,3] / supercell, 1)[1]
            if not in_array_list(gens_span, gen):
                gens_span.extend(_generate_full_symmetry_ops([np.eye(4), gen], (1,1,1)))
                generators_final.append(gen)
                dimensions_final.append(dim)
        return Group(generators_final, dimensions_final)

    @staticmethod
    def getGroupFromSymbol(symbol : str):
        """
        Creates Group object fot given spacegroup symbol.
        :param symbol: Spacegroup symbol.
        :return: Group object.
        """
        generators = [np.asarray(generator) for generator in DECOMPOSITIONS[symbol]['generators']]
        dimensions = DECOMPOSITIONS[symbol]['dimensions']
        return Group(generators, dimensions)


class Subgroups(Sequence):
    """
    Class representing sequence of subgroups.
    Creating list of subgroups is an expensive operation so we simulate such list
    and generate requested subgroup on the fly instead.
    """


    def __init__(self, generators : list, dimensions : list, combinations : list, supercell : tuple):
        """
        Initialize subgroups sequence.
        :param group: Parent group.
        :param combinations: Combinations of generators to be considered.
        """
        self.generators = generators
        self.dimensions = dimensions
        self.combinations = combinations
        self.supercell = supercell
        self.combinationWeights = []
        for sub, rem in self.combinations:
            self.combinationWeights.append(np.asarray(self.dimensions)[np.asarray(rem)].prod() ** (len(sub) - 1))
        self.combinationRanges = np.cumsum(self.combinationWeights)
        self.operators = _generate_full_symmetry_ops(self.generators, self.supercell)


    def __getitem__(self, ind):
        """
        Get a subgroup with index ind.
        :param ind: Index.
        :return: Group object of a subgroup.
        """
        combInd = (self.combinationRanges > ind).nonzero()[0][0]
        ind = ind - self.combinationRanges[combInd - 1] if combInd else ind
        sub_ind, rem_ind = self.combinations[combInd]
        subgroup_generators = [self.generators[i] for i in sub_ind]
        remainder_generators = np.stack(self.generators[i] for i in rem_ind)
        subgroup_dimensions = np.asarray(self.dimensions)[np.asarray(sub_ind)]
        remainder_dimensions = np.asarray(self.dimensions)[np.asarray(rem_ind)]
        dividers = remainder_dimensions.cumprod()
        for i in range(1, len(subgroup_generators)):
            decomposition = np.floor_divide(ind, dividers).tolist()
            decomposition.insert(0, ind)
            ind = decomposition.pop()
            mask = np.remainder(decomposition, remainder_dimensions)
            assert mask[0] == 0, mask
            for order, operator in zip(mask, remainder_generators):
                while order:
                    subgroup_generators[i] = np.dot(subgroup_generators[i], operator)
                    order -= 1
        assert ind == 0
        return Group.getWrapedGroup(subgroup_generators, subgroup_dimensions.tolist(), self.supercell)

    def __len__(self):
        """
        Returns number of subgroups.
        :return: Number of subgroups
        """
        return self.combinationRanges[-1]

    def __call__(self, nodes):
        """
        Calculates orbits of provided positions with respect to this group.
        The orbit of a position is the set of positions which can be obtained by acting with operations of the group.
        :param nodes: array of positions.
        :return: list of orbits, each orbit is an array of positions.
        """
        return [uniqueRows(np.divmod(np.around(orbit, decimals=6), self.supercell)[1], thresh=1.0e-4)  / self.supercell
                for orbit in (np.dot(self.operators[:,:3,:3], nodes.T)
                              + self.operators[:,:3,3].reshape(self.operators.shape[0],3,1)).transpose((2, 0, 1))]
