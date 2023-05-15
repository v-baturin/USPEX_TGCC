"""
USPEX.ConvexHull
================

Class for ConvexHull

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from scipy.spatial import ConvexHull as QHull
from typing import List

logger = logging.getLogger(__name__)


'''
FunctionFolder/USPEX/301/final_convex_hull.m
FunctionFolder/USPEX/301/extendedConvexHull_301.m
FunctionFolder/USPEX/src/CheckDecomposition.m
'''


class Simplex:
    """
    Description of n-dimensional simplex on a given set of points
    """

    def __init__(self, coords):
        self._coords = np.array(coords)
        self.space_dim, self.simplex_dim = self._coords.shape
        self.origin = self._coords[-1]
        if self.space_dim == self.simplex_dim + 1:
            self._aug = np.concatenate([coords, np.ones((self.space_dim, 1))], axis=-1)
            if np.isclose(np.linalg.det(self._aug), 0.0):
                raise ValueError('Error: vertex coordinates are degenerated!')
            self._aug_inv = np.linalg.inv(self._aug)

    def bary_coords(self, point):
        try:
            return np.dot(np.concatenate([point, [1]]), self._aug_inv)
        except AttributeError:
            raise ValueError('Error: simplex is not full-dimensional!')


class ConvexHull(object):
    """
    Class for calculating of convex hull of some data points and retrieving height and depth information.
    """

    def __init__(self, systems: np.ndarray):
        """

        :param systems: array of data points.

        """
        logger.debug(f'ConvexHull arguments: {systems[:,:-1]}, properties: {systems[:,-1]}')
        if not len(systems):
            return

        size = len(systems)
        self._argument = systems[:, :-1]
        self._properties = systems[:, -1]
        self._height = np.full(size, np.inf)
        self._depth = np.full(size, -np.inf)

        properties = self._properties.tolist()
        coords = self._argument.tolist()

        min_ID, max_ID = np.argmin(properties), np.argmax(properties)

        entries_set = np.column_stack((properties, coords))
        # m - number of structures
        # n - size of principal components
        m,n = entries_set.shape


        if n == 1:  # Single component
            min_E, max_E = np.min(properties), np.max(properties)
            for i, prop in enumerate(properties):
                self._height[i] = prop - min_E
                self._depth[i] = prop - max_E
        elif m <= n:    # Not enough point to build proper CH, so all structures are on CH
            for i, (p1, coord1) in enumerate(zip(properties, coords)):
                _dists = []
                for p2, coord2 in zip(properties, coords):
                    if np.allclose(coord1, coord2):
                        _dists.append(p1-p2)
                self._height[i] = np.max(_dists)
                self._depth[i] = np.min(_dists)
        else:
            qhull = QHull(entries_set)

            assert min_ID in qhull.vertices, f'Structure with ID={min_ID} has lowest energy and must be on the CH.'
            assert max_ID in qhull.vertices, f'Structure with ID={max_ID} has highest energy and must be on the CH.'

            simplecies = []
            props = []
            for facet in qhull.simplices:
                try:
                    x = Simplex(np.array([coords[x] for x in facet]))
                    simplecies.append(x)
                    props.append([properties[x] for x in facet])
                except:
                    # This is the case, when Simplex cannot be created due to degenerate input matrix
                    # For composition case it means 2 or more structures have the same composition, but different energies
                    continue

            for i, (p, coord) in enumerate(zip(properties, coords)):
                # This is a dists to simplicies of CH.
                # Positive when distance to the lower bound, negative when distance to the upper bound.
                _dists = []
                for e, s in zip(props, simplecies):
                    y = s.bary_coords(coord)
                    # Filter zeros from the weights. Then we have proper comparison of weights.
                    # if all weights are the same sign - point is somewhere inside or on the border of simplex.
                    y1 = list(filter(lambda x: not np.isclose(x, 0.0), y))
                    if np.abs(np.sum(np.sign(y1))) == len(y1):
                        _dists.append(np.round(p - np.dot(y, e), 6))
                self._height[i] = np.max(_dists)
                self._depth[i] = np.min(_dists)

            # Check whether structure with lowest property is on CH
            for v in qhull.vertices:
                assert np.isclose(self._height[v], 0.0) or np.isclose(self._depth[v], 0.0)

        assert np.isclose(self._height[min_ID], 0.0)
        assert np.isclose(self._depth[max_ID], 0.0)
        # assert set(qhull.vertices) == set(chain(self.lower_bound, self.upper_bound))

    @property
    def lower_bound(self) -> List[int]:
        '''
        :return: IDs of structures, which are on the lower bound of CH
        '''
        return list(i for i, x in enumerate(self._height) if np.isclose(x, 0.0))

    @property
    def upper_bound(self) -> List[int]:
        '''
        :return: IDs of structures, which are on the lower bound of CH
        '''
        return list(i for i, x in enumerate(self._depth) if np.isclose(x, 0.0))

    @property
    def depth(self) -> np.ndarray:
        """
        :return: array of depth of all point below upper bound.
        """
        return self._depth

    @property
    def height(self) -> np.ndarray:
        """
        :return: arrau of heights abve lower bound.
        """
        return self._height


# class ConvexHull_old(object):
#     """
#     Convex hull in space of chemical composition and enthalpy. Dictionary-like object, where keys are structures
#     with defined composition and values are energy above convex hull.
#
#     :examples:
#
#     Create empty convex hull object:
#
#     >>> convex_hull = ConvexHull(config)
#
#     To put system on convex hull:
#
#     >>> convex_hull.add(system)
#
#     To check if energy above convex hull for some system is negative:
#
#     >>> if convex_hull[system] < 0: pass
#
#     To delete system from convex hull:
#
#     >>> del convex_hull[system]
#
#     """
#
#     def __init__(self, config, elements=None):
#         """
#         Initializes the class.
#
#         :type config: :class:`~USPEX.Common.Config.Config` or descendant
#         :param config: describes the chemical compositions configuration space.
#         :type elements: list
#         :param elements: if provided, describes already known structures on convex hull.
#         """
#         self.config = config
#         if elements is not None:
#             self.elements = copy.copy(elements)
#             for check_value in copy.copy(self.elements):
#                 tmp_convex_hull = copy.copy(self)
#                 del tmp_convex_hull[check_value]
#                 if tmp_convex_hull[check_value] > 0:
#                     del self[check_value]
#         else:
#             self.elements = []
#
#     def __getitem__(self, item):
#         """
#         Special method wich allows the '[]' operator. Calculates energy above convex hull,
#         i.e. the minimal formation energy of a given structure regarding all possible decompositions
#         on structures which are on convex hull.
#
#         :type item: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
#         :param item: structure for which we wish to calculate the energy above convex hull.
#         """
#         if not self.elements:
#             return -np.inf
#         elif item in self.elements:
#             return 0
#         else:
#             composition = self.config.numBlocks(composition = item.composition)
#             energy = item.enthalpy/sum(composition)
#
#             composition_hull = []
#             energy_hull = []
#             for system in self.elements:
#                 comp = self.config.numBlocks(composition = system.composition)
#                 composition_hull.append(comp)
#                 energy_hull.append(system.enthalpy/sum(comp))
#
#             N_Comp = len(self.elements)
#             N_Block = min(len(self.config.blocks), N_Comp)
#
#             C = np.zeros((N_Block, len(self.config.blocks)))
#             E = np.zeros(N_Block)
#
#             # here we look through all combinations of systems on convex hull
#             # and calculate formation energy of given system regarding each combination.
#             # combinations restricted to number of blocks: for binary systems -- combinations of two, for ternary --- of three, so on.
#             # when convex hull not yet contain enough systems,
#             # we consider formation energy regarding combination of all systems on convex hull
#             form_Eng = []
#             for comb in combinations(range(N_Comp), N_Block):
#                 for j in range(N_Block):
#                     C[j, :] = composition_hull[comb[j]]
#                     C[j, :] /= np.sum(C[j, :])  # normalization
#                     E[j] = energy_hull[comb[j]]
#
#                 # X represents decomposition of given composition to the compositions of chosen combination.
#                 # all of its elements must be positive because this decomposition have physical meaning of mixture.
#                 try:
#                     X, res, rank, s = np.linalg.lstsq(C.T, composition, rcond=None)
#                     if not res:
#                         res = np.dot(X, C) - composition
#                 except np.linalg.LinAlgError:
#                     continue
#
#                 if np.all(X >= -1.0e-8) and np.linalg.norm(res) < 1.0e-8:
#                     form_Eng.append(energy - np.dot(X, E) / np.sum(composition))  # eV/Block
#
#             if form_Eng:
#                 return max(form_Eng)
#             else:
#                 return -np.inf
#
#     def add(self, key):
#         """
#         Method which puts a structure on convex hull only if its energy above convex hull is nonpositive.
#         """
#         if key in self.elements:
#             return
#         self.elements.append(key)
#         for check_value in copy.copy(self.elements):
#             tmp_convex_hull = copy.copy(self)
#             del tmp_convex_hull[check_value]
#             if tmp_convex_hull[check_value] > 0:
#                 del self[check_value]
#
#     def __delitem__(self, key):
#         """
#         Special methods which allows 'del' operator. Deletes structure from convex hull.
#         """
#         self.elements.remove(key)
#
#     def __copy__(self):
#         """
#         Special method which allow correctly make a copy of current convex hull using copy() operator.
#         """
#         newone = type(self)(self.config)
#         newone.elements = copy.copy(self.elements)
#         return newone
