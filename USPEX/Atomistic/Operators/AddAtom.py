import numpy as np
from scipy.linalg import norm
from copy import deepcopy


class AddAtom:
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
            raise RuntimeError("AddAtom does not currently work in molecular regime.")
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
        coordinates = structure.getCartesianCoordinates()

        newAtomType = np.random.choice(species)

        coordinationNumbers = self.bondHardnessUtility.calcCoordinationNumbers(structure)
        deltaCNs = np.empty(atomTypes.shape, dtype=float)
        for atomType in species:
            inds = (atomTypes == atomType).nonzero()
            atomTypeCNs = coordinationNumbers[inds]
            deltaCNs[inds] = (atomTypeCNs - atomTypeCNs.mean())**2

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

        operations = {newAtomType.short_name: [[newAtomCoords]]}
        offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
        offspring['molecules'][0:0] = molecules
        atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(**offspring)
        minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
        composition = self.simpleMoleculeUtility.composition(offspring)
        if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
            self.environmentUtility.putEnvironment(offspring, environment)
            self.conditions.putConditions(offspring)
            tagsAddRemove[mol1Ind].append(f'added_{newAtomType}')
            tagsAddRemove[mol2Ind].append(f'added_{newAtomType}')
            offspring['tagsAddRemove'] = deepcopy(tagsAddRemove)
            offspring['tagsAddRemove'].append([])
            # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
            # if self.bonds.isConnected(structure):
            return offspring,
