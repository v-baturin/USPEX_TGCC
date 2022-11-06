import numpy as np

_TRANS_ATTEMPTS = 1000


class Transmutation:

    def __init__(self, utilities, howManyTrans = 5, transAttempts = _TRANS_ATTEMPTS):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        # if self.simpleMoleculeUtility.isTrueMolecular:
        #     raise RuntimeError("Transmutation does not currently work in molecular regime.")
        self.specificTrans = []
        self.howManyTrans = howManyTrans
        self.transAttempts = transAttempts

    def __call__(self, system, *args, **kwargs):
        molecules = system['molecules']
        cell = system['cell']
        structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(molecules, cell)
        if self.cellUtility.isGoodCell(cell.getEnvelopeCell(structure.getCartesianCoordinates())):
            symbolsIn = self.simpleMoleculeUtility.moleculeTypes(system)
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
                    position = molecules[i].getCenterOfMassCartesianCoordinates()
                    operation[0:3, 3] = position
                    if s in operations:
                        operations[s].append([[np.copy(operation)]])
                    else:
                        operations[s] = [[[np.copy(operation)]]]

                offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
                offspring['molecules'][0:0] = [molecule for i, molecule in enumerate(molecules) if i not in excluded]
                if 'environment' in system:
                    offspring['environment'] = system['environment']
                atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**offspring)
                minDistMatrix = self.bondUtility.getDistances(atomSymbols, self.conditions.externalPressure)
                if disassembler.environment is not None:
                    inds = disassembler.envIndices
                    atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[tuple(np.meshgrid(inds, inds))]
                composition = self.simpleMoleculeUtility.composition(offspring)
                if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
                    self.conditions.putConditions(offspring)
                    # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
                    # if self.bonds.isConnected(structure):
                    return (offspring,)

        raise RuntimeError("Transmutation failed.")
