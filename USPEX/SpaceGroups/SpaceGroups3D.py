"""
USPEX.Common.SpaceGroups.SpaceGroups3D
======================================

Objects and functions for working with 3D space groups

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import os
import json
import numpy as np

from collections import Sequence
from copy import copy, deepcopy
from itertools import combinations
from pymatgen.symmetry.groups import in_array_list
from scipy.spatial.distance import squareform, pdist

import logging
logger = logging.getLogger(__name__)


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
with open(f'{HOMEPATH}/decompositions.json', 'rt') as f:
    DECOMPOSITIONS = json.load(f)

# These are matrices describing translations along each axis.
# When we consider supercells, we need to add such operations to our generators.
TRANSLATIONS = np.array([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                         [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                         [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]])


def _generate_full_symmetry_ops(generators, supercell: tuple = (1, 1, 1)):
    """
    This function is a modification of the _generate_full_symmetry_ops method in the SpaceGroup class of the module
    pymatgen.symmetry.groups, which is distributed under the terms of the MIT License. Copyright (c) Pymatgen
    Development Team. This function generates an envelope of the set of space group generators. It includes
    translations within supercell.

    :type generators: list
    :param generators: list of 4x4 matrices
    :type supercell: tuple
    :param supercell: three integers, describing the supercell along three dimensions.
    :rtype: numpy array
    :return: full set of symmetry operations obtained from combination of the input generators.
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
    Returns the subset of rows that are unique, in terms of Euclidean distance or otherwise specified metric.

    :type arr: numpy array
    :param arr: array of positions.
    :type thresh: float
    :param thresh: threshold distance
    :type metric: str
    :param metric: the metric to use for computing distance between positions.
    :rtype: numpy array
    :return: unique rows.
    """
    distances = squareform(pdist(arr, metric=metric))
    idxset = {tuple(np.nonzero(v)[0]) for v in distances <= thresh}
    return arr[[x[0] for x in idxset]]


def calcOrbits(operators, nodes, supercell: tuple = (1, 1, 1)):
    """
    Calculates orbits of provided positions with respect to given operators.
    The orbit of a position is the set of positions which can be obtained by acting with operators.

    :type operators: numpy array
    :param operators: N*4*4 array of matrices describing operators which should create an orbit from each position.
    :type nodes: numpy array
    :param nodes: M*3 array of positions.
    :type supercell: tuple
    :param supercell: tuple describing supercell which should be respected when calculating orbits.
    :rtype: list
    :return: list of orbits, each orbit is an array of positions.
    """

    # orbits - M*N*3 array where N - number of operations, M - number of positions
    orbits = (np.dot(operators[:, :3, :3], nodes.T) +
              operators[:, :3, 3].reshape(operators.shape[0], 3, 1)).transpose((2, 0, 1))
    orbits = np.remainder(orbits, supercell)
    orbits[np.where(np.abs(np.asarray(supercell) - orbits) < 1e-4)] = 0

    return list(uniqueRows(orbit, thresh=1.0e-4) / supercell for orbit in orbits)


class Group(object):
    """
    Class representing a space group.
    """

    def __init__(self, generators: list, dimensions: list):
        """
        Initialize Group object.

        :type generators: list
        :param generators: operators which generates our group.
        :type dimensions: list
        :param dimensions: dimensions of provided generators.
        """
        self.generators = copy(generators)
        self.dimensions = copy(dimensions)
        self._operators = _generate_full_symmetry_ops(self.generators)

    @property
    def operators(self):
        """
        :rtype: numpy array
        :return: symmetry operators.
        """
        return self._operators

    def __contains__(self, op):
        """
        Check if this group contains the provided operation.

        :type op: numpy array
        :param op: 4x4 float matrix representing operation.
        :rtype: bool
        :return: True if the group contains the provided operation, False otherwise.
        """
        return in_array_list(self.operators, op)

    def __call__(self, nodes):
        """
        Calculates orbits of provided positions with respect to this group.
        The orbit of a position is the set of positions which can be obtained by acting with operations of the group.

        :type nodes: numpy array
        :param nodes: array of positions.
        :rtype: list
        :return: list of orbits, where each orbit is an array of positions.
        """
        return calcOrbits(self.operators, nodes)

    def redefineGenerators(self):
        """
        If the number of operators in this group mismatches the product of dimensions of generators, this method
        tries to fix it by trying to choose some other set of generators. Such fix is not possible in some cases.
        """

        attempts = 5
        while len(self.operators) != np.prod(self.dimensions) and attempts:
            # First generator is identity. We dont want to consider it.
            generators = self.generators[1:]
            dimensions = self.dimensions[1:]
            for pair in combinations(list(range(len(generators))), 2):
                if pair[0] < len(generators) and pair[1] < len(generators):
                    # We considering all pairs of generators and a group they generates.
                    operators = _generate_full_symmetry_ops([np.eye(4), generators[pair[0]], generators[pair[1]]],
                                                            (1, 1, 1))
                    # If this group dimension mismathches product of dimensions of this two operators
                    # we try to chose other set of generators for this group.
                    if len(operators) != dimensions[pair[0]] * dimensions[pair[1]]:
                        tmpDimensions = [1]
                        for operator in operators[1:]:
                            tmpDimensions.append(len(_generate_full_symmetry_ops([np.eye(4), operator], (1, 1, 1))))
                        # We want to chose as generator the operator with largest dimension.
                        indLarge = np.argsort(tmpDimensions)[-1]
                        dim, rem = np.divmod(len(operators), tmpDimensions[indLarge])
                        assert rem == 0, (len(operators), tmpDimensions[indLarge])
                        if dim == 1:
                            # If the span of this largest generator covers all group then we done
                            # and just delete the second generator. Proceed to actually changing first generator
                            del generators[pair[1]]
                            del dimensions[pair[1]]
                        else:
                            indSmall = None
                            if dim in tmpDimensions:
                                # If there exist operators which dimension equals dimension of the group divided by
                                # dimension of the largest operator then we check such operators if they could
                                # generate the whole group together with largest.
                                for ind in np.flatnonzero(np.asarray(tmpDimensions) == dim):
                                    if len(_generate_full_symmetry_ops([np.eye(4), operators[indLarge],
                                                                        operators[ind]])) == len(operators):
                                        indSmall = ind
                                        # If found we done. Proceed to actually changing first generator
                                        break
                            if indSmall is None:
                                # It is possible that there is a generator with mismatched dimension but which
                                # generates the whole group together with largest. Take it as the second generator.
                                # This does not solves the problem but it makes the generator choice more robust.
                                # Looking for generator based on its dimension starting from second from largest
                                # to the lowest.
                                for dimTry in np.unique(tmpDimensions)[-2:0:-1]:
                                    for ind in np.flatnonzero(np.asarray(tmpDimensions) == dimTry):
                                        if len(_generate_full_symmetry_ops(
                                                [np.eye(4), operators[indLarge], operators[ind]], (1, 1, 1))) \
                                                == len(operators):
                                            indSmall = ind
                                            break
                                    if indSmall is not None:
                                        # Proceed to actually changing first generator
                                        break
                            if indSmall is None:
                                # If we did not find a solution here just proceed to next pair
                                # without actually changing generators.
                                continue
                            # Actually change second generator.
                            generators[pair[1]] = operators[indSmall]
                            dimensions[pair[1]] = tmpDimensions[indSmall]
                        # Actually change first generator.
                        generators[pair[0]] = operators[indLarge]
                        dimensions[pair[0]] = tmpDimensions[indLarge]
                        break
            # Finally add identity
            generators.insert(0, np.eye(4))
            dimensions.insert(0, 1)
            self.generators = generators
            self.dimensions = dimensions
            attempts -= 1
        assert len(_generate_full_symmetry_ops(self.generators, (1, 1, 1))) == len(self._operators)
        if not attempts:
            logger.debug('Number of operators {} mismatches generators dimensions {}'.format(len(self.operators),
                                                                                             self.dimensions))

    def getAllSubgroups(self, supercell: tuple = (1, 1, 1)):
        """
        Enumerates all subgroups of this group which preserve the given supercell.

        :type supercell: tuple
        :param supercell: 3-tuple describing supercell.
        :rtype: :class:`Subgroups`
        :return: Subgroups object.
        """
        generators = copy(self.generators)
        dimensions = copy(self.dimensions)

        extendedOperators = self.operators
        for m, translation in zip(supercell, TRANSLATIONS):
            if m > 1 and not in_array_list(extendedOperators, translation):
                generators.append(translation)
                dimensions.append(m)
                extendedOperators = _generate_full_symmetry_ops(generators, supercell)

        genNum = len(generators)
        inds = []  # Indices of generators preserving supercell.
        for i, generator in enumerate(generators):
            if np.allclose(np.abs(np.dot(generator[0:3, 0:3], supercell)), supercell) \
                    and not np.allclose(generator, np.eye(4)):
                inds.append(i)

        generatorsCombinations = [((0,), tuple(range(genNum)))]
        for q in range(len(inds)):
            for subgroup in combinations(inds, q+1):
                remainder = tuple(set(inds) - set(subgroup))
                generatorsCombinations.append(((0,) + subgroup, (0,) + remainder))
        return Subgroups(generators, dimensions, generatorsCombinations, supercell)

    def getNotPositionInvariantSubgroups(self, position):
        """
        Enumerates all subgroups of this group which preserve the given position.

        :type position: numpy array
        :param position: 3-array describing atomic position.
        :rtype: :class:`Subgroups`
        :return: Subgroups object.
        """
        # First we need to split complex generators like 6-fold axis into product of simpler ones like 3-fold axis
        # and plane. This is needed because some positions may be preserved by one of such simpler generators and
        # not by the other, which make the whole complex generator nontrivial and we get mismatch in number of
        # nontrivial operations and actual multiplicities of positions.
        generators = [np.eye(4)]
        dimensions = [1]
        for generator, dim in zip(self.generators[1:], self.dimensions[1:]):
            group = Group([np.eye(4), generator], [1, dim])
            positions = group(np.asarray([position]))[0]
            generators.append(generator)
            dimensions.append(dim)
            if len(positions) != len(group.operators):
                trivialExists = False
                for possible_generator in group.operators:
                    subgroup = Group([np.eye(4), possible_generator], [1, dim])
                    if (len(subgroup.operators) == len(positions)) \
                            and (len(subgroup(np.asarray([position]))[0]) == len(positions)):
                        generators[-1] = possible_generator
                        dimensions[-1] = len(subgroup.operators)
                        trivialExists = True
                        break
                if trivialExists:
                    for possible_generator in group.operators[1:]:
                        if np.allclose(np.mod(np.around(np.dot(possible_generator[0:3, 0:3], position)
                                                        + possible_generator[0:3, 3], decimals=6), (1, 1, 1)),
                                       position, atol=1e-4):
                            subgroup = Group([np.eye(4), possible_generator], [1, dim])
                            generators.append(possible_generator)
                            dimensions.append(len(subgroup.operators))

        # We want to divide our generators into two parts. One -- generators preserving our position.
        # And second -- ones not preserving.
        # Generators preserving our position might differ from our initial generators.
        allGenerators = [np.identity(4, dtype=np.float)]
        allDimensions = [1]
        trivialGenerators = [0]
        nonTrivialGenerators = [0]

        for generator, dimension in zip(generators, dimensions):
            if not np.allclose(generator, np.identity(4, dtype=np.float)):
                operators = _generate_full_symmetry_ops(allGenerators, (1, 1, 1))
                trivial = False
                for modifier in self.operators:
                    # Modify generator and make sure its translation part lies between 0 and 1.
                    # And if close to 1 then substitute with 0.
                    modifiedGenerator = np.dot(modifier, generator)
                    modifiedGenerator[0:3, 3] = np.mod(modifiedGenerator[0:3, 3], (1, 1, 1))
                    modifiedGenerator[np.where(np.abs(np.asarray((1, 1, 1)) - modifiedGenerator[0:3, 3]) < 1e-4), 3] = 0
                    # Check if this generator presumes our position
                    if np.allclose(np.mod(np.around(np.dot(modifiedGenerator[0:3, 0:3], position)
                                                    + modifiedGenerator[0:3, 3], decimals=6),
                                          (1, 1, 1)), position, atol=1e-4):
                        # Check if it is already in the group formed by processed generators.
                        if not in_array_list(operators, modifiedGenerator):
                            # Check if modification did not changed its dimension
                            span = _generate_full_symmetry_ops([np.eye(4), modifiedGenerator], (1, 1, 1))
                            if len(span) == dimension:
                                # If all checks passes add this modified generator to list of processed generators
                                # and to list of trivial generators.
                                allGenerators.append(modifiedGenerator)
                                allDimensions.append(dimension)
                                trivialGenerators.append(len(allGenerators) - 1)
                                trivial = True
                                break
                if not trivial:
                    # If generator can not be modified in a way it preserves our position
                    # then it should be added to the list of nontruvial generators. And to list of processed generators.
                    allGenerators.append(generator)
                    allDimensions.append(dimension)
                    nonTrivialGenerators.append(len(allGenerators) - 1)

        # If we missed some generator we should add it as nontrivial.
        # This code probably is not needed but checking it demand a lot of work. Will do it later.
        # TODO check if this is needed.
        operators = _generate_full_symmetry_ops(allGenerators, (1, 1, 1))
        for generator, dimension in zip(generators, dimensions):
            if not in_array_list(operators, generator):
                allGenerators.append(generator)
                allDimensions.append(dimension)
                nonTrivialGenerators.append(len(allGenerators) - 1)
        # We check if number of operators match product of dimensions of our generators. If not we try to drop some.
        # Actually it might be not needed any more. But checking this demand a lot of testing. I will get here later
        # TODO consider removing this.
        operators = _generate_full_symmetry_ops(allGenerators, (1, 1, 1))
        if len(operators) != np.prod(allDimensions):
            n = 1
            while n < len(nonTrivialGenerators):
                i = nonTrivialGenerators[n]
                tmpGenerators = copy(allGenerators)
                tmpDimensions = copy(allDimensions)
                del tmpGenerators[i]
                del tmpDimensions[i]
                tmpOperators = _generate_full_symmetry_ops(tmpGenerators)
                if (len(operators) == len(tmpOperators)) and (len(operators) == np.prod(tmpDimensions)):
                    allGenerators = tmpGenerators
                    allDimensions = tmpDimensions
                    tmpNonTrivialGenerators = []
                    for j in nonTrivialGenerators:
                        if j < i:
                            tmpNonTrivialGenerators.append(j)
                        elif j > i:
                            tmpNonTrivialGenerators.append(j-1)
                    nonTrivialGenerators = tmpNonTrivialGenerators
                    tmpTrivialGenerators =[]
                    for j in trivialGenerators:
                        if j < i:
                            tmpTrivialGenerators.append(j)
                        elif j > i:
                            tmpTrivialGenerators.append(j-1)
                    trivialGenerators = tmpTrivialGenerators
                else:
                    n += 1

        return Subgroups(allGenerators, allDimensions, [(nonTrivialGenerators, trivialGenerators)])

    @staticmethod
    def getWrappedGroup(generators: list, dimensions: list, supercell: tuple):
        """
        Creates wrapped group. Supercell becomes cell.

        :type generators: list
        :param generators: operators which generates our group.
        :type dimensions: list
        :param dimensions: dimensions of provided generators.
        :type supercell: tuple
        :param supercell: 3-tuple describing supercell.
        :rtype: :class:`Group`
        :return: wrapped group.
        """
        generators = deepcopy(generators)
        generators_final = [np.eye(4)]
        dimensions_final = [1]
        operators = _generate_full_symmetry_ops(generators_final, (1, 1, 1))
        for gen in generators:
            gen[0:3, 3] = np.divmod(gen[0:3, 3] / supercell, 1)[1]
            gen[np.where(np.abs(np.asarray((1, 1, 1)) - gen[0:3, 3]) < 1e-4), 3] = 0
            if not in_array_list(operators, gen):
                span = _generate_full_symmetry_ops([np.eye(4), gen], (1, 1, 1))
                generators_final.append(gen)
                dimensions_final.append(len(span))
                operators = _generate_full_symmetry_ops(generators_final)
        group = Group(generators_final, dimensions_final)
        group.redefineGenerators()
        return group

    @staticmethod
    def getGroupFromSymbol(symbol: str):
        """
        Creates Group object fot given spacegroup symbol.

        :type symbol: str
        :param symbol: spacegroup symbol.
        :rtype: :class:`Group`
        :return: space group corresponding to the given symbol.
        """
        generators = [np.asarray(generator) for generator in DECOMPOSITIONS[symbol]['generators']]
        dimensions = DECOMPOSITIONS[symbol]['dimensions']
        return Group(generators, dimensions)


class Subgroups(Sequence):
    """
    Class representing sequence of subgroups.
    Creating a list of subgroups is an expensive operation, so we simulate such list
    and generate the requested subgroup on the fly instead.
    """

    def __init__(self, generators: list, dimensions: list, combinations: list, supercell: tuple = (1, 1, 1)):
        """
        Initialize subgroups sequence.

        :type generators: list
        :param generators: operators which generates our group.
        :type dimensions: list
        :param dimensions: dimensions of provided generators.
        :type combinations: list
        :param combinations: combinations of generators to be considered.
        :type supercell: tuple
        :param supercell: 3-tuple describing supercell.
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

    def __getitem__(self, ind: int):
        """
        Get a subgroup with index ind.

        :type ind: int
        :param ind: index.
        :rtype: :class:`Group`
        :return: the requested subgroup.
        """
        logger.debug('Trying {}th subgroup'.format(ind))
        # First we determine combination of generators in which range ind gets.
        combInd = (self.combinationRanges > ind).nonzero()[0][0]
        # Now redefine ind as index within subgroups corresponding deterined combination.
        ind = ind - self.combinationRanges[combInd - 1] if combInd else ind
        # Get generators and their dimensions which got to the subgroup (subgroup_generators and subgroup_dimensions)
        # and those which did not get to it but will be used to modify it (remainder_generators and remainder_dimensions).
        sub_ind, rem_ind = self.combinations[combInd]
        subgroup_generators = [self.generators[i] for i in sub_ind]
        remainder_generators = np.stack(tuple(self.generators[i] for i in rem_ind))
        subgroup_dimensions = np.asarray(self.dimensions)[np.asarray(sub_ind)]
        remainder_dimensions = np.asarray(self.dimensions)[np.asarray(rem_ind)]
        # We interpret our ind as set of masks for each generators got to the subgroup defining which operations
        # from remainder are used to modify it. Mask looks like set of numbers for each generator from remainder
        # defining how much times this generator should be used to modify currents generator from the subgroup.
        # Such numbers should not exceed dimensions of respective remainder generators.
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
        return Group.getWrappedGroup(subgroup_generators, subgroup_dimensions.tolist(), self.supercell)

    def __len__(self):
        """
        Returns number of subgroups.

        :rtype: int
        :return: Number of subgroups
        """
        # Our subgroups divided into blocks according combinations of generators. Boundaries of these blocks are stored
        # in combinationRanges. So the total number of our subgroups correspond to the last boundary.
        return self.combinationRanges[-1]

    def calcOrbits(self, nodes):
        """
        Calculates orbits of provided positions with respect to operators of this Subgroups object and its supercell.
        The orbit of a position is the set of positions which can be obtained by acting with operations of the group.

        :type nodes: numpy array
        :param nodes: array of positions.
        :rtype: list
        :return: list of orbits, where each orbit is an array of positions.
        """
        return calcOrbits(self.operators, nodes, self.supercell)
