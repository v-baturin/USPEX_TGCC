import numpy as np

_TRANS_ATTEMPTS = 1000


class Transmutation:

    def __init__(self, utilities, howManyTrans = 5, transAttempts = _TRANS_ATTEMPTS):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("Transmutation does not currently work in molecular regime.")
        self.specificTrans = []
        self.howManyTrans = howManyTrans
        self.transAttempts = transAttempts

    def __call__(self, system, *args, **kwargs):
        molecules = system['molecules']
        cell = system['cell']
        environment = system['environment'] if 'environment' in system else None
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            symbolsIn = self.simpleMoleculeUtility.moleculeTypes(system)
            symbolsOut = self.compositionSpace.symbols

            trans = np.array([(i,sOut) for i, sIn in enumerate(symbolsIn) for sOut in symbolsOut if sIn != sOut],
                             dtype = [('index', int),('symbol', 'U10')])

            for _ in range(self.transAttempts):
                numberOfTrans = np.random.randint(1, self.howManyTrans + 1)
                permutation = np.random.choice(trans, numberOfTrans)
                transCoordinates = {}
                excluded = []
                for i, s in permutation:
                    excluded.append(i)
                    if s in transCoordinates:
                        transCoordinates[s].append([molecules[i].getCenterOfMassCartesianCoordinates()])
                    else:
                        transCoordinates[s] = [[molecules[i].getCenterOfMassCartesianCoordinates()]]

                offspring = self.simpleMoleculeUtility.populateStructure(cell, transCoordinates, None)
                offspring['molecules'][0:0] = [molecule for i, molecule in enumerate(molecules) if i not in excluded]
                atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(**offspring)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                composition = self.simpleMoleculeUtility.composition(offspring)
                if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
                    self.environmentUtility.putEnvironment(offspring, environment)
                    self.conditions.putConditions(offspring)
                    return (offspring,)

        raise RuntimeError("Transmutation failed.")
