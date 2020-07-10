"""
USPEX.Common.Atomistic.ConvexHull
=================================

Class for ConvexHull

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


'''
FunctionFolder/USPEX/301/final_convex_hull.m
FunctionFolder/USPEX/301/extendedConvexHull_301.m
FunctionFolder/USPEX/src/CheckDecomposition.m
'''

import logging
logger = logging.getLogger(__name__)


import numpy as np
import os
import pandas as pd

from copy import copy
from itertools import chain, combinations
from scipy.spatial import ConvexHull as QHull
from typing import List, Tuple



class Simplex:
    '''
    Description of n-dimensional simplex on a given set of points
    '''

    def __init__(self, coords):
        self._coords = np.array(coords)
        self.space_dim, self.simplex_dim = self._coords.shape
        self.origin = self._coords[-1]
        if self.space_dim == self.simplex_dim + 1:
            self._aug = np.concatenate([coords, np.ones((self.space_dim, 1))], axis=-1)
            self._aug_inv = np.linalg.inv(self._aug)

    def bary_coords(self, point):
        try:
            return np.dot(np.concatenate([point, [1]]), self._aug_inv)
        except AttributeError:
            raise ValueError('Error: simplex is not full-dimensional!')


class ConvexHull:

    # ID of systems on the convex hull
    # _systems = []
    # all_IDs = []

    def __init__(self, saved_data : str, property:str='enthalpy_per_atom'):
        self._df = pd.DataFrame(columns=['system', 'principal_component', 'property', 'height', 'depth'])
        self.property = property

        # self._df = pd.DataFrame(columns=['ID', 'enthalpy_per_atom', 'fingerprint', 'height'])
        self.SAVED_DATAFILE = saved_data
        self.systems = []

    def extend(self, systems):
        if not len(systems):
            return
        # if os.path.exists(self.SAVED_DATAFILE):
        #     self._df = pd.read_pickle(self.SAVED_DATAFILE)

        self.systems.extend(systems)
        for i, system in enumerate(self.systems):
            self._df.loc[i] = system, system.principal_component, getattr(system, self.property), np.inf, -np.inf

        properties = self._df.property.tolist()
        coords = self._df.principal_component.tolist()

        min_ID, max_ID = np.argmin(properties), np.argmax(properties)

        principal_component_set = np.column_stack((properties, coords))
        # m - number of structures
        # n - size of principal components
        m,n = principal_component_set.shape


        if n == 1:  # Single component
            min_E, max_E = np.min(properties), np.max(properties)
            for i, row in self._df.iterrows():
                self._df.at[i, 'height'] = row.property - min_E
                self._df.at[i, 'depth'] = row.property - max_E
        elif m <= n:    # Not enough point to build proper CH, so all structures are on CH
            for p1, coord1, (i, _) in zip(properties, coords, self._df.iterrows()):
                _dists = []
                for p2, coord2, (j, _) in zip(properties, coords, self._df.iterrows()):
                    if np.allclose(coord1, coord2):
                        _dists.append(p1-p2)
                self._df.at[i, 'height'] = np.max(_dists)
                self._df.at[i, 'depth'] = np.min(_dists)
        else:
            qhull = QHull(principal_component_set)

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

            for p, coord, (i, _) in zip(properties, coords, self._df.iterrows()):
                # This is a dists to simplicies of CH.
                # Positive when distance to the lower bound, negative when distance to the upper bound.
                _dists = []
                for e, s in zip(props, simplecies):
                    y = s.bary_coords(coord)
                    if np.abs(np.sum(np.sign(y))) == len(y) or np.any(np.isclose(y, 0.0)):
                        _dists.append(np.round(p - np.dot(y, e), 6))
                self._df.at[i, 'height'] = np.max(_dists)
                self._df.at[i, 'depth'] = np.min(_dists)

            # Check whether structure with lowest property is on CH
            for v in qhull.vertices:
                assert np.isclose(self._df.iloc[v].height, 0.0) or np.isclose(self._df.iloc[v].depth, 0.0)

        assert np.isclose(self._df.iloc[min_ID].height, 0.0)
        assert np.isclose(self._df.iloc[max_ID].depth, 0.0)
        # assert set(qhull.vertices) == set(chain(self.lower_bound, self.upper_bound))

        # self._df.to_pickle(self.SAVED_DATAFILE)

    def clean(self):
        if os.path.exists(self.SAVED_DATAFILE):
            os.remove(self.SAVED_DATAFILE)

    @property
    def lower_bound(self) -> list:
        '''
        :return: IDs of structures, which are on the lower bound of CH
        '''
        return list(x.system for _,x in self._df.iterrows() if np.isclose(x.height, 0.0))

    @property
    def upper_bound(self) -> list:
        '''
        :return: IDs of structures, which are on the lower bound of CH
        '''
        return list(x.system for _,x in self._df.iterrows() if np.isclose(x.depth, 0.0))

    @property
    def depth(self):
        return self._df.depth.to_numpy()

    @property
    def height(self):
        return self._df.height.to_numpy()

    # def depth(self, ID : int):
    #     '''
    #     Calculates depth below the top of convex hull
    #     : param ID: ID of the structure
    #     '''
    #     for _, row in self._df.iterrows():
    #         if ID == row.ID:
    #             return row.depth
    #     return -np.inf
    #
    # def height(self, ID : int):
    #     '''
    #     Calculates height above convex hull
    #     : param ID: ID of the structure
    #     '''
    #     for _, row in self._df.iterrows():
    #         if ID == row.ID:
    #             return row.height
    #     return np.inf


DEFAULT_DATA_FILE = 'compositionCH.dump'


class CompositionConvexHull(ConvexHull):
    """
    Convex hull in space of chemical composition and enthalpy. Dictionary-like object, where keys are structures
    with defined composition and values are energy above convex hull.

    :examples:

    Create empty convex hull object:
    >>> convex_hull = CompositionConvexHull(config)

    To extend convex hull by set of systems:
    >>> convex_hull.extend(systems)
    """


    def __init__(self, config, saved_data:str=None):
        """
        :type config: :class:`~USPEX.Common.Config.Config` or descendant
        :param config: describes the chemical compositions configuration space.
        """
        if saved_data is None:
            saved_data = DEFAULT_DATA_FILE
        super().__init__(saved_data=saved_data, property='enthalpy_per_block')
        self.config = config

    def extend(self, systems):
        for system in systems:
            composition = self.config.numBlocks(system.composition)
            system.enthalpy_per_block = system.enthalpy/sum(composition)
            system.principal_component = composition[:-1] / np.sum(composition)
        super().extend(systems)


class ConvexHull_old(object):
    """
    Convex hull in space of chemical composition and enthalpy. Dictionary-like object, where keys are structures
    with defined composition and values are energy above convex hull.

    :examples:

    Create empty convex hull object:

    >>> convex_hull = ConvexHull(config)

    To put system on convex hull:

    >>> convex_hull.add(system)

    To check if energy above convex hull for some system is negative:

    >>> if convex_hull[system] < 0: pass

    To delete system from convex hull:

    >>> del convex_hull[system]

    """

    def __init__(self, config, elements=None):
        """
        Initializes the class.

        :type config: :class:`~USPEX.Common.Config.Config` or descendant
        :param config: describes the chemical compositions configuration space.
        :type elements: list
        :param elements: if provided, describes already known structures on convex hull.
        """
        self.config = config
        if elements is not None:
            self.elements = copy.copy(elements)
            for check_value in copy.copy(self.elements):
                tmp_convex_hull = copy.copy(self)
                del tmp_convex_hull[check_value]
                if tmp_convex_hull[check_value] > 0:
                    del self[check_value]
        else:
            self.elements = []

    def __getitem__(self, item):
        """
        Special method wich allows the '[]' operator. Calculates energy above convex hull,
        i.e. the minimal formation energy of a given structure regarding all possible decompositions
        on structures which are on convex hull.

        :type item: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
        :param item: structure for which we wish to calculate the energy above convex hull.
        """
        if not self.elements:
            return -np.inf
        elif item in self.elements:
            return 0
        else:
            composition = self.config.numBlocks(item.composition)
            energy = item.enthalpy/sum(composition)

            composition_hull = []
            energy_hull = []
            for system in self.elements:
                comp = self.config.numBlocks(system.composition)
                composition_hull.append(comp)
                energy_hull.append(system.enthalpy/sum(comp))

            N_Comp = len(self.elements)
            N_Block = min(len(self.config.blocks), N_Comp)

            C = np.zeros((N_Block, len(self.config.blocks)))
            E = np.zeros(N_Block)

            # here we look through all combinations of systems on convex hull
            # and calculate formation energy of given system regarding each combination.
            # combinations restricted to number of blocks: for binary systems -- combinations of two, for ternary --- of three, so on.
            # when convex hull not yet contain enough systems,
            # we consider formation energy regarding combination of all systems on convex hull
            form_Eng = []
            for comb in combinations(range(N_Comp), N_Block):
                for j in range(N_Block):
                    C[j, :] = composition_hull[comb[j]]
                    C[j, :] /= np.sum(C[j, :])  # normalization
                    E[j] = energy_hull[comb[j]]

                # X represents decomposition of given composition to the compositions of chosen combination.
                # all of its elements must be positive because this decomposition have physical meaning of mixture.
                try:
                    X, res, rank, s = np.linalg.lstsq(C.T, composition, rcond=None)
                    if not res:
                        res = np.dot(X, C) - composition
                except np.linalg.LinAlgError:
                    continue

                if np.all(X >= -1.0e-8) and np.linalg.norm(res) < 1.0e-8:
                    form_Eng.append(energy - np.dot(X, E) / np.sum(composition))  # eV/Block

            if form_Eng:
                return max(form_Eng)
            else:
                return -np.inf

    def add(self, key):
        """
        Method which puts a structure on convex hull only if its energy above convex hull is nonpositive.
        """
        if key in self.elements:
            return
        self.elements.append(key)
        for check_value in copy.copy(self.elements):
            tmp_convex_hull = copy.copy(self)
            del tmp_convex_hull[check_value]
            if tmp_convex_hull[check_value] > 0:
                del self[check_value]

    def __delitem__(self, key):
        """
        Special methods which allows 'del' operator. Deletes structure from convex hull.
        """
        self.elements.remove(key)

    def __copy__(self):
        """
        Special method which allow correctly make a copy of current convex hull using copy() operator.
        """
        newone = type(self)(self.config)
        newone.elements = copy.copy(self.elements)
        return newone
