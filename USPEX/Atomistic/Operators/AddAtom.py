import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.spatial.distance import cdist
from ..AtomicPrimitives import AtomicStructure
from ..Element import Element


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
        self.availableAtomsDatabase = None

    def __call__(self, system):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        environment = system['environment'] if 'environment' in system else None
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)  # ,environment)
        atomTypes = structure.getAtomTypes()
        uniqueAtomTypes = np.unique(atomTypes)
        coordinates = structure.getCartesianCoordinates()

        coordinationNumbers = self.bondHardnessUtility.calcCoordinationNumbers(structure)
        deltaCNs = np.empty(atomTypes.shape, dtype=float)
        for atomType in uniqueAtomTypes:
            inds = (atomTypes == atomType).nonzero()
            atomTypeCNs = coordinationNumbers[inds]
            deltaCNs[inds] = (atomTypeCNs - atomTypeCNs.mean())**2
        i = np.random.choice(len(structure), p=deltaCNs/deltaCNs.sum())
        j = i + 1

        atom1Type = atomTypes[i]
        atom2Type = atomTypes[j]
        newAtomType = np.random.choice(uniqueAtomTypes)
        # for molecules we should estimate its radius instead of using covalent
        newBondLength = newAtomType.covalent_radius + np.max([atom1Type.covalent_radius, atom2Type.covalent_radius])

        massCenter = structure.getCenterOfMassFractionalCoordinates()
        atom1coord = coordinates[i]
        atom2coord = coordinates[j]

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


    @classmethod
    def createAtomDatabase(cls, structure, cell):
        # 'atomInd': atom index
        # 'availability': 1 - this position hasn't been used
        #                 0 - this position has been used (not available)
        #                 -1 - doesn't belong to surface: for future alpha-surfaces code
        # !availability column for each type of atom/molecule to add
        # !availability for atom to remove
        species = np.unique(structure.getAtomTypes())
        Natoms = len(structure.getAtomTypes())
        columns = ['atomType', 'coordNum', 'removability'] + ['addability|{}'.format(i) for i in species]
        availAtomDB = pd.DataFrame(np.ones((Natoms, 3 + len(species))), columns=columns)
        for atomInd in range(Natoms):
            atomType = structure.getAtomTypes()[atomInd]
            availAtomDB.loc[atomInd, 'atomType'] = atomType
            availAtomDB.loc[atomInd, 'coordNum'] = cls.calculate_coordination_numbers(atomType.covalent_radius, cell,
                                                                                      structure.getCartesianCoordinates()[Natoms])
