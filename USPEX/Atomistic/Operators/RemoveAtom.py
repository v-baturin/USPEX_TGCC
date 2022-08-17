import numpy as np
from copy import copy, deepcopy


class RemoveAtom:
    def __init__(self, utilities):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.ionDistances = utilities.ionDistances
        self.bonds = utilities.bonds
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        self.bondHardnessUtility = utilities.bondHardnessUtility
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("RemoveAtom does not currently work in molecular regime.")
        self.availableAtomsDatabase = None

    def __call__(self, system):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        environment = system['environment'] if 'environment' in system else None
        if 'tagsAddRemove' not in system:
            system['tagsAddRemove'] = [[] for _ in range(len(molecules))]
        tagsAddRemove = system['tagsAddRemove']
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)  # ,environment)
        atomTypes = structure.getAtomTypes()
        species = np.unique(atomTypes)

        coordinationNumbers = self.bondHardnessUtility.calcCoordinationNumbers(structure)
        deltaCNs = np.empty(atomTypes.shape, dtype=float)
        for atomType in species:
            inds = (atomTypes == atomType).nonzero()
            atomTypeCNs = coordinationNumbers[inds]
            deltaCNs[inds] = (atomTypeCNs - atomTypeCNs.mean())**2

        for _ in range(100):
            i = np.random.choice(len(structure), p=deltaCNs/deltaCNs.sum())
            molInd = disassembler.findMolIndex(i)
            if 'removed' not in tagsAddRemove[molInd]:
                break
        else:
            raise RuntimeError("RemoveAtom failed.")

        offspring = {'molecules': [], 'cell': cell}
        for molecule, inds in zip(molecules, disassembler.indices):
            if i not in inds:
                offspring['molecules'].append(copy(molecule))

        atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(**offspring)
        minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
        composition = self.simpleMoleculeUtility.composition(offspring)
        if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
            self.environmentUtility.putEnvironment(offspring, environment)
            self.conditions.putConditions(offspring)
            tagsAddRemove[molInd].append('removed')
            offspring['tagsAddRemove'] = deepcopy(tagsAddRemove)
            del offspring['tagsAddRemove'][molInd]
            # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
            # if self.bonds.isConnected(structure):
            return offspring,
