import numpy as np
from scipy.linalg import norm
from copy import deepcopy


class AddAtom:
    def __init__(self, utilities):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.compositionSpace = utilities.compositionSpace
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.cellUtility = utilities.cellUtility
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("AddAtom does not currently work in molecular regime.")
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
        coordinates = structure.getCartesianCoordinates()

        coordinationNumbers = self.bondUtility.calcCoordinationNumbers(structure)
        deltaCNs = np.empty(atomTypes.shape, dtype=float)
        for atomType in species:
            inds = (atomTypes == atomType).nonzero()
            atomTypeCNs = coordinationNumbers[inds]
            deltaCNs[inds] = (atomTypeCNs - atomTypeCNs.mean())**2

        for attempt in range(100):

            newAtomType = np.random.choice(species)

            for _ in range(100):
                i = np.random.choice(len(structure), p=deltaCNs/deltaCNs.sum())
                atom1Type = atomTypes[i]
                atom1coord = coordinates[i]
                mol1Ind = disassembler.findMolIndex(i)
                if f'added_{newAtomType}' not in tagsAddRemove[mol1Ind]:
                    break
            else:
                raise RuntimeError("AddAtom failed.")

            edges = []
            coef = 1.4
            while not edges:
                for j, (atomType, coord) in enumerate(zip(atomTypes, coordinates)):
                    dist = norm(coord - atom1coord)
                    if 0 < dist <= coef*(atom1Type.covalent_radius + atomType.covalent_radius):
                        edges.append(j)
                coef *= 1.1

            for _ in range(100):
                j = np.random.choice(edges)
                atom2Type = atomTypes[j]
                atom2coord = coordinates[j]
                mol2Ind = disassembler.findMolIndex(j)
                if f'added_{newAtomType}' not in tagsAddRemove[mol2Ind]:
                    break
            else:
                raise RuntimeError("AddAtom failed.")

            # TODO for molecule we should estimate its radius instead of using covalent
            newBondLength = newAtomType.covalent_radius + np.max([atom1Type.covalent_radius, atom2Type.covalent_radius])

            massCenter = structure.getCenterOfMassFractionalCoordinates()

            edgeCenter = 0.5*(atom1coord + atom2coord)
            edgeVector = atom1coord - atom2coord
            edgeLength = norm(edgeVector)
            edgeVector /= edgeLength
            if edgeLength/2 > newBondLength:
                newAtomCoords = edgeCenter
            else:
                vectorInPlain = edgeCenter - massCenter
                normal = vectorInPlain - np.dot(vectorInPlain, edgeVector) * edgeVector
                normal /= norm(normal)
                newAtomCoords = edgeCenter + np.sqrt(newBondLength**2 - (edgeLength/2)**2) * normal + 0.01*np.random.rand(3)

            operation = np.eye(4, dtype=float)
            operation[0:3, 3] = cell.cartesianToFractional(newAtomCoords)
            operations = {newAtomType.short_name: [[[operation]]]}
            offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
            offspring['molecules'][0:0] = molecules
            if 'environments' in system:
                offspring['environments'] = system['environments']
            offspring = offspringFactory(**offspring)
            structure = offspring.getAtomicStructure()
            minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                          self.conditions.externalPressure)
            if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix) \
                    and self.compositionSpace.isGoodComposition(self.simpleMoleculeUtility.composition(offspring)):
                self.conditions.putConditions(offspring)
                tagsAddRemove[mol1Ind].append(f'added_{newAtomType}')
                tagsAddRemove[mol2Ind].append(f'added_{newAtomType}')
                offspring.setProperty('tagsAddRemove', deepcopy(tagsAddRemove))
                offspring['tagsAddRemove'].append([])
                # if self.bonds.isConnected(structure):
                return offspring,

        raise RuntimeError("AddAtom failed.")
