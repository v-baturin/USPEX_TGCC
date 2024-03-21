import numpy as np
from copy import copy, deepcopy


class RemoveAtom:
    def __init__(self, utilities, suffix):
        self.atomistic = utilities.atomistic
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        self.suffix = suffix
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("RemoveAtom does not currently work in molecular regime.")
        self.availableAtomsDatabase = None

    def __call__(self, system, offspringFactory=None):
        molecules = system.getProperty('molecules', extension='atomistic', suffix=self.suffix)
        cell = system.getProperty('cell', extension='atomistic', suffix=self.suffix)
        environments = system.getProperty('environments', extension='atomistic', suffix=self.suffix)
        if 'addRemove.tags.origin' not in system:
            system.setProperty('tags', [[] for _ in range(len(molecules))], extension='addRemove', suffix='origin')
        tagsAddRemove = system.getProperty('tags', extension='addRemove', suffix='origin')
        structure, disassembler = self.atomistic.atomicDisassemblerType.assemble({'atomistic.molecules': molecules,
                                                                                  'atomistic.cell': cell})
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

            offspring = {'atomistic.molecules': copy(molecules), 'atomistic.cell': cell}
            del offspring['atomistic.molecules'][molInd]
            offspring['atomistic.environments'] = environments
            offspring = offspringFactory(**offspring)
            offspringAtomTypes = offspring.getProperty('structure', extension='atomistic').getAtomTypes()
            minDistMatrix = self.bondUtility.getDistances(offspringAtomTypes, self.conditions.externalPressure)
            if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix) \
                    and self.compositionSpace.isGoodComposition(offspring.getProperty('composition', extension='simpleMoleculeUtility')):
                self.conditions.putConditions(offspring)
                tagsAddRemove[molInd].append('removed')
                system.setProperty('tags', tagsAddRemove, extension='addRemove', suffix='origin')
                tagsAddRemove_offspring = deepcopy(tagsAddRemove)
                del tagsAddRemove_offspring[molInd]
                offspring.setProperty('tags', tagsAddRemove_offspring, extension='addRemove')
                return offspring,

        raise RuntimeError("RemoveAtom failed.")
