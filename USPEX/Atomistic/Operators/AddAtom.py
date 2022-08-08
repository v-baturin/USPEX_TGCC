import numpy as np
import pandas as pd
from scipy.linalg import norm


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
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)  # ,environment)
        atomTypes = structure.getAtomTypes()
        species = np.unique(atomTypes)
        coordinates = structure.getCartesianCoordinates()

        coordinationNumbers = self.bondHardnessUtility.calcCoordinationNumbers(structure)
        deltaCNs = np.empty(atomTypes.shape, dtype=float)
        for atomType in species:
            inds = (atomTypes == atomType).nonzero()
            atomTypeCNs = coordinationNumbers[inds]
            deltaCNs[inds] = (atomTypeCNs - atomTypeCNs.mean())**2
        i = np.random.choice(len(structure), p=deltaCNs/deltaCNs.sum())
        atom1Type = atomTypes[i]
        atom1coord = coordinates[i]

        edges = []
        coef = 1.4
        while not edges:
            for atomType, coord in zip(atomTypes, coordinates):
                dist = norm(coord - atom1coord)
                if 0 < dist <= coef*(atom1Type.covalent_radius + atomType.covalent_radius):
                    edges.append((atomType, coord))
            coef *= 1.1
        atom2Type, atom2coord = np.random.choice(edges)

        newAtomType = np.random.choice(species)
        # for molecules we should estimate its radius instead of using covalent
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

        # new structure must be added to database

        operations = {newAtomType.short_name: [[newAtomCoords]]}
        offspring = self.simpleMoleculeUtility.populateStructure(cell, operations)
        offspring['molecules'][0:0] = molecules
        atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(**offspring)
        minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
        composition = self.simpleMoleculeUtility.composition(offspring)
        if np.all(atomDistances >= minDistMatrix) and self.compositionSpace.isGoodComposition(composition):
            self.environmentUtility.putEnvironment(offspring, environment)
            self.conditions.putConditions(offspring)
            # structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(**offspring)
            # if self.bonds.isConnected(structure):
            return offspring,


    def createAtomDatabase(self, structure):
        # 'atomInd': atom index
        # 'availability': 1 - this position hasn't been used
        #                 0 - this position has been used (not available)
        #                 -1 - doesn't belong to surface: for future alpha-surfaces code
        # !availability column for each type of atom/molecule to add
        # !availability for atom to remove
        atomTypes = structure.getAtomTypes()
        species = np.unique(atomTypes)
        coordinationNumbers = self.bondHardnessUtility.calcCoordinationNumbers(structure)
        columns = ['atomType', 'coordNum', 'removability'] + [f'addability|{atomType}' for atomType in species]
        availAtomDB = pd.DataFrame(np.ones((len(structure), 3 + len(species))), columns=columns)
        for atomInd, atomType in enumerate(atomTypes):
            availAtomDB.loc[atomInd, 'atomType'] = atomType
            availAtomDB.loc[atomInd, 'coordNum'] = coordinationNumbers[atomInd]
