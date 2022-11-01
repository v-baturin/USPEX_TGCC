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
        if 'tagsAddRemove' not in system:
            system['tagsAddRemove'] = [[] for _ in range(len(molecules))]
        tagsAddRemove = system['tagsAddRemove']
        structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(molecules, cell)  # ,environment)
        atomTypes = structure.getAtomTypes()
        species = np.unique(atomTypes)

        coordinationNumbers = self.bondHardnessUtility.calcCoordinationNumbers(structure)
        deltaCNs = np.empty(atomTypes.shape, dtype=float)
        for atomType in species:
            inds = (atomTypes == atomType).nonzero()
            atomTypeCNs = coordinationNumbers[inds]
            deltaCNs[inds] = (atomTypeCNs - atomTypeCNs.mean())**2

        for attempt in range(100):
            for _ in range(100):
                i = np.random.choice(len(structure), p=deltaCNs/deltaCNs.sum())
                molInd = disassembler.findMolIndex(i)
                if 'removed' not in tagsAddRemove[molInd]:
                    break
            else:
                raise RuntimeError("RemoveAtom failed.")

            offspring = {'molecules': [], 'cell': cell}
            offspring['molecules'][0:0] = molecules
            del offspring['molecules'][molInd]
            atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**offspring)
            minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
            if disassembler.environment is not None:
                inds = disassembler.envIndices
                atomDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[
                    tuple(np.meshgrid(inds, inds))]
            composition = self.simpleMoleculeUtility.composition(offspring)
            if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
                if 'environment' in system:
                    offspring['environment'] = system['environment']
                self.conditions.putConditions(offspring)
                tagsAddRemove[molInd].append('removed')
                offspring['tagsAddRemove'] = deepcopy(tagsAddRemove)
                del offspring['tagsAddRemove'][molInd]
                # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
                # if self.bonds.isConnected(structure):
                return offspring,

        raise RuntimeError("RemoveAtom failed.")
