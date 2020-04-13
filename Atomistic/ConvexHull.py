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

import itertools
import numpy as np
import copy


class ConvexHull(object):
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
            for comb in itertools.combinations(range(N_Comp), N_Block):
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
