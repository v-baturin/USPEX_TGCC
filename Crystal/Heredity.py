import logging
logger = logging.getLogger(__name__)

'''
@file        Heredity.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        15 November 2016
@brief       Heredity variation operator.
'''


import numpy as np
from copy import copy

from ..Atomistic.calcDefaultVolume import calcVolumeForComposition
from ..Atomistic.CompositionSpace import Composition
from ..VarOperator import VarOperator, VOFailed

'''
FunctionFolder/USPEX/3**/Heredity_3**.m
FunctionFolder/USPEX/3**/heredity(_final)(_molecule)(_var).m
'''

MAX_ATTEMPS = 50


class Heredity(VarOperator):


    def __init__(self, systemFactory, config, pool, utilities):
        '''
        :param initFrac : float - initial fraction of population to be generated with heredity
        :param minFrac : float - minimal fraction of population to be generated with heredity
        :param config: reference to configuration space object
        '''
        super(Heredity, self).__init__(systemFactory, config, pool, utilities)
        self.compositionSpace = utilities['compositionSpace']
        # assert 'percSliceShift' in params and isinstance(params['percSliceShift'], float)
        # self._percSliceShift = params['percSliceShift']
        self.correlationFO = 0        # fitness-order correlation

    def _makeRandomSlab(self, system, dimension, fracMin : float, fracMax : float):
        lat = system.get_cell()
        if dimension == 0:
            L = abs(system.get_volume() / np.linalg.norm(np.cross(lat[1, :], lat[2, :])))
        elif dimension == 1:
            L = abs(system.get_volume() / np.linalg.norm(np.cross(lat[0, :], lat[2, :])))
        else:
            L = abs(system.get_volume() / np.linalg.norm(np.cross(lat[0, :], lat[1, :])))

        Lchar = 0.5 * (system.get_volume() / len(system)) ** (1 / 3)
        # characteristic length - approximate 'radius' of the atom in the cell = 0.5 * (V / N) ^ 1 / 3

        N = int(round(L / (Lchar + (L - Lchar) * (np.cos(self.correlationFO * np.pi / 2)) ** 2)))

        child_good, child_bad, order_good, order_bad = make_slab(system, dimension, fracMin, fracMax, np.random.random(3))
        good_slab_aorder_ref = 0 if self.correlationFO <= 0 else np.max(system.order)
        for i in range(1, N):
            slabs = make_slab(system, dimension, fracMin, fracMax, np.random.random(3))
            good_slab_order = slabs[2]
            good_slab_aorder = sum(good_slab_order) / len(good_slab_order) if good_slab_order else good_slab_aorder_ref
            if (good_slab_aorder - good_slab_aorder_ref) * self.correlationFO <= 0:
                good_slab_aorder_ref = good_slab_aorder
                child_good, child_bad, order_good, order_bad = slabs
        return child_good, child_bad, order_good, order_bad

    def tune(self, population : list):
        # TODO calculate correlation Fitness - Order
        order = []
        enthalpy = []
        for system in population:
            order.append(system.averageOrder)
            enthalpy.append(system.enthalpy)
        self.correlationFO = np.corrcoef(order,enthalpy)[0,1]
        if np.isnan(self.correlationFO):
            self.correlationFO = 0

    def __call__(self, system1, system2, dimension=None, fracFrac=None, fracShift1=None, fracShift2=None,
                 desiredComposition = None, fracLattice=None) -> tuple:
        '''
        The heredity was designed to allow structural heritage. 
        This function looks terribly complicated - but don't worry, it's all
        quite straight forward.
        Keep in mind that we are dealing with a periodical problem (periodicity : 1)
        => example (coordinate) : 2.41 = 1.41 = 0.41 = -0.59 = - 1.59...
        '''

        logger.debug(f'Heredity: system {system1.ID} and system {system2.ID}, correlation coefficient {self.correlationFO}')

        if dimension is None or fracFrac is None or fracShift1 is None or fracShift2 is None\
                or desiredComposition is None or fracLattice is None:
            attempts_left = MAX_ATTEMPS
        else:
            attempts_left = 1
        while attempts_left:
            # choose dimension in which parents will be sliced
            if dimension is None:
                dimension = np.random.randint(0, 3)

            # fracFrac determines what spatial fraction of the one parent will be taken.
            # The rest is taken from the other parent. Thus: 0.25 means one spatial
            # fourth versus three fourths.
            if fracFrac is None:
                fracFrac = 0.25 + np.random.rand() * 0.5

            # TODO perSliceShift

            # make slabs from parents. each parent gives two slubs: good one and bad one.
            # good one is supposed to have better local order
            if fracShift1 is None:
                child_good1, child_bad1, order_good1, order_bad1 = self._makeRandomSlab(system1, dimension, 0, fracFrac)
            else:
                child_good1, child_bad1, order_good1, order_bad1 = make_slab(system1, dimension, 0, fracFrac, fracShift1)
            if fracShift2 is None:
                child_good2, child_bad2, order_good2, order_bad2 = self._makeRandomSlab(system2, dimension, fracFrac, 1)
            else:
                child_good2, child_bad2, order_good2, order_bad2 = make_slab(system2, dimension, fracFrac, 1, fracShift2)

            # child_good and child_bad are lists of molecules
            child_good = child_good1 + child_good2
            child_bad = child_bad1 + child_bad2
            # order_good and order_bad are lists of local orders of respective molecules in parent structures
            order_good = order_good1 + order_good2
            order_bad = order_bad1 + order_bad2

            # sort child molecule lists according their order. good one from worst to best, bad one from best to worst
            if self.correlationFO <= 0:
                child_good = [child_good[i] for i in np.argsort(order_good)[:]]
                child_bad = [child_bad[i] for i in np.argsort(order_bad)[::-1]]
            else:
                child_good = [child_good[i] for i in np.argsort(order_good)[::-1]]
                child_bad = [child_bad[i] for i in np.argsort(order_bad)[:]]

            # calculate total composition of molecules in good_child
            composition = np.zeros(len(self.compositionSpace.symbols), dtype = float)
            for molecule in child_good:
                composition += self.compositionSpace.numIons(molecule.composition)

            # determine desired composition of ofspring structure. this function is nondeterministic.
            if desiredComposition is None:
                desiredComposition = self.compositionSpace.findDesiredComposition(
                    system1.composition, system2.composition, composition)[0]

            if desiredComposition is None:
                attempts_left -= 1
                continue

            # lettice of offspring is a linear combination of parents' lattices with coefficient fracLattice
            if fracLattice is None:
                fracLattice = np.random.rand()

            temp_potLat = fracLattice * system1.get_cell() + (1 - fracLattice) * system2.get_cell()
            volLat = np.linalg.det(temp_potLat)
            if volLat < 0:
                temp_potLat = -1 * temp_potLat

            # scale the lattice to the volume we assume it approximately to be
            if system1.get_chemical_formula() == system2.get_chemical_formula():
                latVol = system1.get_volume() * fracFrac + system1.get_volume() * (1 - fracFrac)
            else:
                # latVol = np.dot(desiredComposition, self.config.calcVolume())
                latVol = calcVolumeForComposition(
                    Composition(dict(zip(self.compositionSpace.symbols, desiredComposition)),
                                self.compositionSpace.moleculesTypeToFormula), **self.config)
            potentialLattice = temp_potLat * (latVol / volLat) ** (1.0 / 3.0)

            # remove extra molecules from child_good structure
            child_good_new = []
            for i, molecule in enumerate(child_good):
                molIndex = self.compositionSpace.symbols.index(molecule.molSymbol[0])
                current = composition[molIndex]
                desired = desiredComposition[molIndex]
                if current > desired:
                    composition -= self.compositionSpace.numIons(molecule.composition)
                else:
                    child_good_new.append(molecule)
            child_good = child_good_new

            # add lacking molecules to good_child from bad_child
            for symbol, current, desired in zip(self.compositionSpace.symbols, composition, desiredComposition):
                needed = desired - current
                if needed > 0:
                    for molecule in child_bad:
                        if molecule.molSymbol[0] == symbol:
                            child_good.append(molecule)
                            needed -= 1
                            if needed == 0:
                                break

            child_good = self.systemFactory(molecules=child_good, cell=potentialLattice, optimizeLattice=True, **self.config)

            if child_good.isGoodSystem() and self.compositionSpace.isGoodComposition(child_good.composition):
                crystal = child_good
                self.pool.assignID(crystal)
                crystal.howCome = self.__class__.__name__
                crystal.parent = str(system1.ID) + ' ' + str(system2.ID)
                logger.info(f"Structure {crystal.ID} created parents {system1.ID} and {system2.ID}")
                return (crystal,)

            attempts_left -= 1

        raise VOFailed

def make_slab(system, dimension, fracMin : float, fracMax : float, fracShift : np.ndarray):
    order = system.order
    system = copy(system)
    offset = np.dot(system.get_cell(), fracShift)
    system.translate(offset)
    molecules = system.molecules
    child1 = []
    child2 = []
    order1 = []
    order2 = []
    for molecule, ord in zip(molecules, order):
        offset = np.floor(molecule.get_center_of_mass(scaled=True))
        offset = np.dot(molecule.get_cell(), offset)
        molecule.translate(-offset)
        fracCoordinate = molecule.get_center_of_mass(scaled=True)[dimension]
        if fracCoordinate > fracMin and fracCoordinate < fracMax:
            child1.append(molecule)
            order1.append(ord)
        else:
            child2.append(molecule)
            order2.append(ord)
    return child1, child2, order1, order2


'''
This code was in matlab version, but was never used.
It estimates volume of child structure using volume of structures on convex hull.

if System.blocks.shape[0] == 2:
    N_T = len(numMols)
    ch = POP_STRUC['convex_hull']
    # works only for 2 types of atoms!
    x = numMols(-1) / np.sum(numMols)
    i = 1
    while x > ch[[i + 1], N_T] / np.sum(ch[[i + 1], 0:N_T]):
        i += 1

    v1 = np.linalg.det(POP_STRUC['POPULATION'][ch[[i + 1], N_T + 2]]['LATTICE'])
    v2 = np.linalg.det(POP_STRUC['POPULATION'][ch[[i], N_T + 2]]['LATTICE'])
    a1 = ch[[i + 1], N_T] / np.sum(ch[[i + 1], 0:N_T])
    a2 = ch[[i], N_T] / np.sum(ch[[i], 0:N_T])
    latVol = np.abs((x - a2) / (a2 - a1)) * v1 + np.abs((x - a1) / (a2 - a1)) * v2
'''
