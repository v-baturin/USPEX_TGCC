'''
@file        ConvexHull.py
@author:     Pavel Bushlanov
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        04 September 2017
@brief       Class for ConvexHull
'''


'''
FunctionFolder/USPEX/301/final_convex_hull.m
FunctionFolder/USPEX/301/extendedConvexHull_301.m
FunctionFolder/USPEX/src/CheckDecomposition.m
'''

import itertools
import numpy as np
import copy


class ConvexHull(object):


    def __init__(self,config, elements=None):
        self.config = config
        if elements:
            self.elements = copy.copy(elements)
            for check_value in copy.copy(self.elements):
                tmp_convex_hull = copy.copy(self)
                del tmp_convex_hull[check_value]
                if tmp_convex_hull[check_value] > 0:
                    del self[check_value]
        else:
            self.elements = []

    def __getitem__(self, item):
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
            N_Block = min(len(self.config.blocks),N_Comp)

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
                    X, res, rank, s = np.linalg.lstsq(C.T,composition)
                    if not res:
                        res = np.dot(X,C) - composition
                except np.linalg.LinAlgError:
                    continue

                if np.all(X >= -1.0e-8) and np.linalg.norm(res) < 1.0e-8:
                    form_Eng.append(energy - np.dot(X, E) / np.sum(composition))  # eV/Block

            if form_Eng:
                return max(form_Eng)
            else:
                return -np.inf

    def add(self, key):
        if key in self.elements:
            return
        self.elements.append(key)
        for check_value in copy.copy(self.elements):
            tmp_convex_hull = copy.copy(self)
            del tmp_convex_hull[check_value]
            if tmp_convex_hull[check_value] > 0:
                del self[check_value]

    def __delitem__(self, key):
        self.elements.remove(key)

    def __copy__(self):
        newone = type(self)(self.config)
        newone.elements = copy.copy(self.elements)
        return newone

