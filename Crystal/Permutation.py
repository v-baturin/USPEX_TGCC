import logging
logger = logging.getLogger(__name__)

import numpy as np

from copy import copy
from itertools import combinations
from typing import List, Tuple

from ..VarOperator import VarOperator, VOFailed

__author__='alex_a_marjewski'


class Permutation(VarOperator):

    name = 'Permutation'

    def __init__(self, systemFactory, config, pool, initFrac : float=0.0, howManySwaps : int=5, specificSwaps : List[Tuple[str, str]]=None, minFrac : float=0.1, maxFrac : float=1.0):
        '''
        :param initFrac : float - initial fraction of population to be generated with permutation
        :param minFrac : float - minimal fraction of population to be generated with permutation
        :param config: reference to configuration space object
        :param howManySwaps : int - maximal number of swaps at one operator run
        :param specificSwaps : list - only specific molecular types should be swapped; for Mo <-> B permutation, input
         list should be either ('Mo', 'B') or ('B', 'Mo') - both will work.
        '''
        super(Permutation, self).__init__(systemFactory, config, pool, initFrac, minFrac, maxFrac)
        self.correlationFO = 0        # fitness-order correlation | Ignoring this temporarily
        self.howManySwaps = howManySwaps
        if specificSwaps is not None:
            for s in specificSwaps:
                assert len(set(s)) == 2, f'{s} has been specified, which is wrong. The 2 elements must be set'
            self.specificSwaps = specificSwaps
        else:
            self.specificSwaps = []

    def tune(self, population : list):
        # UNDER CONSTRUCTION
        pass

    def __call__(self, system) -> tuple:

        logger.debug(f'Permutation: system {system.ID}, correlation coefficient {self.correlationFO}')
        crystal_tuple = self.permutation(system)
        return crystal_tuple

    def _assign_data(self, target, ID:int):
        '''
        Assignation of data to the output structure
        '''
        target.howCome = self.name
        self.pool.assignID(target)
        target.parent = str(ID)
        logger.info(f"Structure {target.ID} formed by permutation from {target.parent}")

    def permutation(self, system):

        # Initialize for creation of possible permutations list - create list of indices of molecules
        # Here we obtain all possible permutations
        ms = system.molSymbol
        # And here we remove permutations of molecules of one type
        permutations = [(i1,i2) for (i1,i2) in combinations(range(len(system.molecules)), 2) if ms[i1] != ms[i2]]

        # Here we apply specificSwaps constraint, if such is present.
        if self.specificSwaps:
            permutations = [(s1,s2) for (s1,s2) in permutations if (ms[s1], ms[s2]) in self.specificSwaps or (ms[s2], ms[s1]) in self.specificSwaps]

        #  And then we shuffle the permutation list and enforce 'list' type of data
        permutations = np.random.permutation(permutations).tolist()

        atLeastOnePermutation = False
        try:
            while len(permutations):
                atLeastOnePermutation = False
                # Create a work copy of molecules in our system
                attempted_structure = copy(system.molecules)

                # It's boring to permute molecules just one time. Let's do it from 1 to 'howManySwaps' times! Default howManySwaps is 5.
                for _ in range(np.random.randint(self.howManySwaps)):
                    # Get first permutation from the total list of permutations; then permute
                    s1, s2 = permutations.pop(0)
                    # Generate translation vector for molecules (geometric center of molecule 1 minus geometric center of molecule 2,
                    # it's regular linear algebra).
                    translation_1to2 = attempted_structure[s1].get_center_of_mass(scaled=True) - \
                                       attempted_structure[s2].get_center_of_mass(scaled=True)
                    # Translate molecule 1 to achieve coincidence of it's geometric center with geometric center of molecule 2
                    attempted_structure[s1].translate_scaled(translation_1to2)
                    # ...and the other way around.
                    attempted_structure[s2].translate_scaled(-translation_1to2)
                    atLeastOnePermutation = True

                # Here we generate a complete system
                target = self.systemFactory(molecules=attempted_structure, cell=system.cell,
                                            optimizeLattice=True, **self.config)
                if atLeastOnePermutation and target.isGoodSystem():
                    self._assign_data(target=target, ID=system.ID)
                    return target,
        except IndexError:
            if atLeastOnePermutation and target.isGoodSystem():
                self._assign_data(target=target, ID=system.ID)
                return target,
        logger.info(f"Permutation failed on {system.ID}: exhausted possible permutations. "
                    f"This error may also indicate an attempt to apply permutation to a structure"
                    f" with a single type of atom or molecule.")
        raise VOFailed
