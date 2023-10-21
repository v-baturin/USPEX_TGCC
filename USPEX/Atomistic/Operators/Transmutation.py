import numpy as np

_TRANS_ATTEMPTS = 1000


class Transmutation:

    def __init__(self, utilities, suffix='4', howManyTrans = 5, transAttempts = _TRANS_ATTEMPTS):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        self.suffix = suffix
        # if self.simpleMoleculeUtility.isTrueMolecular:
        #     raise RuntimeError("Transmutation does not currently work in molecular regime.")
        self.specificTrans = []
        self.howManyTrans = howManyTrans
        self.transAttempts = transAttempts

    def __call__(self, system, offspringFactory=None):
        molecules = system.getProperty('molecules', extension='atomistic', suffix=self.suffix)
        cell = system.getProperty('cell', extension='atomistic', suffix=self.suffix)
        structure = system.getProperty('structure', extension='atomistic', suffix=self.suffix)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            symbolsIn = system['simpleMoleculeUtility.moleculeTypes.origin']
            symbolsOut = self.compositionSpace.symbols

            trans = np.array([(i,sOut) for i, sIn in enumerate(symbolsIn) for sOut in symbolsOut if sIn != sOut],
                             dtype = [('index', int),('symbol', 'U10')])

            for _ in range(self.transAttempts):
                numberOfTrans = np.random.randint(1, self.howManyTrans + 1)
                permutation = np.random.choice(trans, numberOfTrans)
                operations = {}
                operation = np.eye(4, dtype=float)
                excluded = []
                for i, s in permutation:
                    excluded.append(i)
                    position = cell.cartesianToFractional(molecules[i].getCenterOfMassCartesianCoordinates())
                    operation[0:3, 3] = position
                    if s in operations:
                        operations[s].append([[np.copy(operation)]])
                    else:
                        operations[s] = [[[np.copy(operation)]]]

                offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
                offspring['atomistic.molecules'][0:0] = [molecule for i, molecule in enumerate(molecules) if i not in excluded]
                offspring = offspringFactory(**offspring)
                try:
                    offspring.setProperty('environments',
                                          system.getProperty('environments', extension='atomistic', suffix=self.suffix),
                                          extension='atomistic')
                except Exception:
                    pass
                structure = offspring.getProperty('structure', extension='atomistic')
                minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                              self.conditions.externalPressure)
                if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix):
                    self.conditions.putConditions(offspring)
                    # if self.bonds.isConnected(structure):
                    return (offspring,)

        raise RuntimeError("Transmutation failed.")
