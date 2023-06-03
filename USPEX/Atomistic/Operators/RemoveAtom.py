import numpy as np
from copy import copy, deepcopy


class RemoveAtom:
    def __init__(self, utilities):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("RemoveAtom does not currently work in molecular regime.")
        self.availableAtomsDatabase = None

    def __call__(self, system, offspringFactory=None):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        if 'tagsAddRemove' not in system:
            system['tagsAddRemove'] = [[] for _ in range(len(molecules))]
        tagsAddRemove = system['tagsAddRemove']
        structure, disassembler = offspringFactory.atomicDisassemblerType.assemble(molecules, cell)  # ,environment)
        atomTypes = structure.getAtomTypes()
        species = np.unique(atomTypes)

        coordinationNumbers = self.bondUtility.calcCoordinationNumbers(structure)
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
            if 'environments' in system:
                offspring['environments'] = system['environments']
            offspring = offspringFactory(**offspring)
            structure = offspring.getAtomicStructure()
            minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                          self.conditions.externalPressure)
            if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix) \
                    and self.compositionSpace.isGoodComposition(self.simpleMoleculeUtility.composition(offspring)):
                self.conditions.putConditions(offspring)
                tagsAddRemove[molInd].append('removed')
                offspring.setProperty('tagsAddRemove', deepcopy(tagsAddRemove))
                del offspring['tagsAddRemove'][molInd]
                # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
                # if self.bonds.isConnected(structure):
                return offspring,

        raise RuntimeError("RemoveAtom failed.")
