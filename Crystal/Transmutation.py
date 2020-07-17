import logging
logger = logging.getLogger(__name__)
import numpy as np

from ..VarOperator import VarOperator, VOFailed

__author__='alex_a_marjewski'

class Transmutation(VarOperator):

    def __init__(self, systemFactory, config, pool, utilities, howManyTrans : int=2, specificTrans : list=[]):
        '''
        :param initFrac : float - initial fraction of population to be generated with permutation
        :param minFrac : float - minimal fraction of population to be generated with permutation
        :param config: reference to configuration space object
        :param howManyTrans : int - maximal number of swaps at one operator run
        :param specificTrans : list - only specific molecular type transmutations are allowed; for Mo<->B transmutations, input
         list should be either ['Mo', 'B'] or ['B', 'Mo'] - both will work.
        '''
        super(Transmutation, self).__init__(systemFactory, config, pool, utilities)
        self.compositionSpace = utilities['compositionSpace']
        self.correlationFO = 0        # fitness-order correlation | Ignoring this temporarily
        self.howManyTrans = howManyTrans
        self.specificTrans = [set(i) for i in specificTrans]

    def tune(self, population : list):
        # UNDER CONSTRUCTION
        pass

    def __call__(self, system) -> tuple:

        logger.debug(f'Transmutation: system {system.ID}, correlation coefficient {self.correlationFO}')
        crystal_tuple = self.transmutation(system)
        return crystal_tuple

    def transmutation(self, system):
        # Initialize for creation of possible transmutation list
        # Each transmutation has the form of a list [index, 'molSymbol']
        indices = list(range(0, len(system.molecules)))
        transmutations = [ [i, j] for i in indices for j in self.compositionSpace.symbols if system.molSymbol[i] != j ]

        # Here we apply specificTrans constraint, if such is present.
        if len(self.specificTrans)>0:
            transmutations = [i for i in transmutations if {system.molSymbol[i[0]], i[1]} in self.specificTrans]

        while True:

            atLeastOneTransmutation = False

            # Create a work copy of molecules in our system
            attempted_structure = system.molecules

            # It's boring to transmute molecules just one time. Let's do it from 1 to 'howManyTrans' times! Default howManyTrans is 2.
            for i in list(range(1, np.random.randint(1, self.howManyTrans) + 1)):

                if not transmutations:
                    if atLeastOneTransmutation:
                        break
                    else:
                        logger.info(f"Transmutation failed on {system.ID}: exhausted possible transmutations. "
                                    f"This error may also indicate an attempt to apply transmutation to a structure"
                                    f" with a single unique molecule/atom type.")
                        raise VOFailed


                # Get first permutation from the total list of permutations; calculate geometric center
                # of the transmuted molecule
                transmute = transmutations.pop(0)
                molecular_geometric_center = attempted_structure[transmute[0]].get_center_of_mass(scaled=True)

                try:
                    target = self.systemFactory(molecules=attempted_structure, cell=system.cell,
                                                optimizeLattice = True, **self.config)

                    molecule_reference = self.systemFactory.fromDICT(self.compositionSpace.molecules[transmute[1]])
                    molecule_reference.set_cell(system.cell)
                    molecule_reference.rotate((360 * np.random.random_sample()), 'z')
                    theta = np.arcsin(np.sqrt(np.random.random_sample())) * 180 / np.pi
                    if np.random.randint(2):
                        theta = 180 - theta
                    molecule_reference.rotate(theta, 'y')
                    molecule_reference.rotate((360 * np.random.random_sample()), 'z')

                    translation_to_desired_position = molecule_reference.get_center_of_mass(scaled = True)\
                                                      - molecular_geometric_center

                    molecule_reference.translate_scaled(translation_to_desired_position)

                    target.removeMolecule(transmute[0])
                    target.extend(molecule_reference)

                except KeyError:
                    target = self.systemFactory(molecules=attempted_structure, cell=system.cell,
                                                optimizeLattice=True, **self.config)
                    target.removeMolecule(transmute[0])
                    target.extend(self.systemFactory(symbols=[transmute[1]], cell=system.cell,
                                                     scaled_positions=[molecular_geometric_center]))

                atLeastOneTransmutation = True

            # Here we generate a complete system
            
            if target.isGoodSystem():
                target.howCome = self.__class__.__name__
                self.pool.assignID(target)
                target.parent = str(system.ID)
                logger.info(f"Structure {target.ID} formed by transmutation from {target.parent}")
                return target,

