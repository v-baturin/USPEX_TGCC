import numpy as np
import pandas as pd
from scipy.linalg import norm
from scipy.spatial.distance import cdist
from ..AtomicPrimitives import AtomicStructure


class AddAtom:
    def __init__(self, utilities):
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.environmentUtility = utilities.environmentUtility
        self.cellUtility = utilities.cellUtility
        self.availableAtomsDatabase = None

    def __call__(self, system):
        ID = system['ID']
        molecules = system['molecules']
        cell = system['cell']
        environment = system['environment'] if 'environment' in system else None
        structure, disassembler = self.simpleMoleculeUtility.structureType.assemble(molecules, cell)  # ,environment)

        # choose it with surface + database utility, type(atom1) = class AtomicStructure
        atom1, atom2 = AtomicStructure(['Si'], np.array([[0,0,0]])), \
                       AtomicStructure(['Si'], np.array([[0,0,2.5]]))
        atom1coord = atom1.getCartesianCoordinates()
        atom2coord = atom2.getCartesianCoordinates()
        newAtomType = 'Si'  # how is chosen? If molecule?
        covRad1 = atom1.getAtomTypes()[0].covalent_radius
        covRad2 = atom2.getAtomTypes()[0].covalent_radius
        covRadNew = newAtomType.covalent_radius
        massCenter = structure.getCenterOfMassFractionalCoordinates()  # of all structure?
        vectorInPlain = 0.5*(atom1coord + atom2coord) - massCenter
        surfaceVector = atom1coord-atom2coord
        if norm(surfaceVector) > covRad1 + covRad2 + 2*covRadNew:
            # adding atom between atom1 and atom2
            # If molecule?
            newAtomCoords = 0.5*(atom1coord + atom2coord)
        else:
            normalVector = vectorInPlain - np.dot(vectorInPlain, surfaceVector)/norm(surfaceVector)**2 * surfaceVector
            newBonDLength = covRadNew + np.max([covRad1, covRad2])
            newAtomCoords = 0.5*(atom1coord + atom2coord) +\
                            (norm(newBonDLength)**2 - norm(0.5*surfaceVector)**2)**0.5 * normalVector +\
                            0.01*np.random.rand(3)
        newStructure = np.concatenate(structure.getCartesianCoordinates(), np.reshape(newAtomCoords, (3,1)), axis=0)
        # to molecule?
        # new structure must be added to database
        return AtomicStructure(structure.getAtomTypes()+newAtomType, newStructure)

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

    @staticmethod
    def calculate_coordination_numbers(radii, lattice, coordinates):
        # radii - covalent radius
        
        # this program calculates coordination numbers of cell with given lattice, coordinates and radii of atoms
        # developer - Bushlanov Pavel
        sz = 27
        closest = np.array(
            [[[0, 0, 0]], [[-1, 0, 0]], [[-1, 0, -1]], [[-1, -1, -1]], [[-1, -1, 0]], [[0, -1, 0]], [[0, -1, -1]],
             [[0, 0, -1]], [[-1, -1, 1]], [[-1, 0, 1]], [[-1, 1, 1]], [[-1, 1, 0]], [[-1, 1, -1]], [[0, -1, 1]],
             [[0, 0, 1]], [[0, 1, 1]], [[0, 1, 0]], [[0, 1, -1]], [[1, 0, 0]], [[1, 0, -1]], [[1, -1, -1]],
             [[1, -1, 0]], [[1, -1, 1]], [[1, 0, 1]], [[1, 1, 1]], [[1, 1, 0]], [[1, 1, -1]]])
        vertices = np.dot(np.concatenate((closest + coordinates), axis=0), lattice)
        radii_large = np.tile(radii, sz)
        dists = cdist(vertices, vertices)
        dists_no_self = np.delete(np.triu(dists, 1), 0, 1) + np.delete(np.tril(dists, -1), dists.shape[1] - 1, 1)
        base_bond_legth = radii_large.reshape(1, radii_large.size) + radii_large.reshape(radii_large.size, 1)
        base_bond_legth_no_self = np.delete(np.triu(base_bond_legth, 1), 0, 1) + np.delete(np.tril(base_bond_legth, -1),
                                                                                           base_bond_legth.shape[1] - 1,
                                                                                           1)
        order = np.exp(-(dists_no_self - base_bond_legth_no_self) / 0.23)
        coord_number = order.sum(axis=1) / order.max(axis=1)
        return coord_number
