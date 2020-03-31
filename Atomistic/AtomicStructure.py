'''
@file        AtomicStructure.py
@author:     Artem Samtsevich
@copyright:  2018 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        16 November 2016
@brief       Class Atoms-type structure with properties adn without periodicity (not Crystal)
'''


__author__ = 'asamtsevich'


from ase.atoms import Atoms

import copy
import json
import numpy as np

from ..System import System
from .optLattice import optLattice
from .Element import Element
from .mol.coord2Zmatrix import coord2Zmatrix
from .mol.zmatrix2coord import zmatrix2coord
from .mol.find_pair import find_pair


class AtomicStructure(System):
    '''
    Class describing generic Atoms-type structure with properties

    '''

    # Appers here because of nonsupport of rational atomic numbers (cuted bonds, H-bonds, etc.)

    bonds = None
    _dielectricTensor = None
    energy = np.inf
    enthalpy = np.inf
    externalPressure = 1.0e-4
    forces = None
    _pressureTensor = None
    dimension = None

    def __init__(self, molecules=[], symbols=None, positions=None, scaled_positions=None, cell=None, pbc=None, info=None,
                 optimizeLattice=False, magmoms=None):
        isSymbols = symbols is not None
        assert (not isSymbols) or (not len(molecules))     # Cannot initialize with both

        self._chemicalSymbols = []
        self._molecules = []
        self.format = []
        self.flex_dihedral = []
        self.molSymbol = []

        if isSymbols:
            for i, s in enumerate(symbols):
                if not isinstance(s, str):
                    s = s.symbol
                if s in ['H.5', 'H.75', 'H1.25', 'H1.5']:
                    symbols[i] = 'H'
                self._chemicalSymbols.append(str(s))
            self.atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions,
                                           positions=positions, pbc=pbc, cell=cell, info=info)
            if optimizeLattice:
                self.optimizeLattice()
            self._molecules = [[i] for i in range(len(symbols))]
            self.format = [[[0,0,0]] for s in symbols]
            self.flex_dihedral = [[] for s in symbols]
            self.molSymbol = copy.copy(self._chemicalSymbols)
        else:
            self.atoms = Atoms(cell=cell, pbc=pbc)
            if optimizeLattice:
                self.optimizeLattice()
            for molecule in molecules:
                self.extend(molecule)

        super().__init__()

    def __getattr__(self, item):
        '''
        Special method for forwarding attributes which are not explicitly defined in AtomicStructure to ase.atoms class.

        :param item: Attribute name
        '''
        if item == '__setstate__' or item == 'atoms' or not hasattr(self, 'atoms'):
            raise AttributeError(self,item)
        elif hasattr(self.atoms, item):
            return getattr(self.atoms, item)
        else:
            raise AttributeError(self,item)

    def __len__(self):
        '''
        Special method which returns number of atoms in the structure.
        '''
        return len(self.atoms)

    def extend(self, ext):
        '''
        The method extends current system with given one.
        This is done by adding molecules from given structure to current one.
        Geometry of all molecules is kept. Cell parameters of current structure is kept.
        New molecules will have same scaled coordinates of their centers as in given structure.

        :param ext: AtomicStructure to extend the current one.
        '''
        for inds, frmt, flex_dihedral, molSymbol in zip(ext._molecules, ext.format, ext.flex_dihedral, ext.molSymbol):
            inds = (np.asarray(inds) + len(self.atoms)).tolist()
            self._molecules.append(inds)
            self.format.append(frmt)
            self.flex_dihedral.append(flex_dihedral)
            self.molSymbol.append(molSymbol)

        self._chemicalSymbols.extend(ext.get_chemical_symbols())
        atoms = copy.copy(ext.atoms)
        molCenter = atoms.get_center_of_mass(scaled=True)
        oldCell = atoms.get_cell()
        newCell = self.get_cell()
        atoms.translate(-np.dot(molCenter,oldCell))
        atoms.set_cell(newCell)
        atoms.translate(np.dot(molCenter, newCell))
        self.atoms.extend(atoms)

    def __add__(self, other):
        '''
        Special method for supporting '+' operator on AtomicStructure. 
        Creates a copy of the first argument and extends it with the second then returns resulting structure.

        :param other: AtomicStructure to be added to the current one.
        '''
        res = copy.copy(self)
        res.extend(other)
        return res

    def __getitem__(self, item):
        '''
        Special method for retrieving an atom from structure.

        :param item: index of an atom to be retrieved.
        :return: ase.Atom object describing an atom with given index.
        '''
        return self.atoms.__getitem__(item)

    @property
    def molecules(self):
        '''
        Property-method which represents current structure as a list of molecules.
        Each molecule is a AtomicStructure object.
        '''
        mols = []
        for inds, frmt, flex_dihedral, molSymbol in zip(self._molecules, self.format, self.flex_dihedral, self.molSymbol):
            mol = AtomicStructure(symbols=self[inds], cell=self.get_cell())
            if len(mol) > 1:
                mol.merge(list(range(len(mol))), frmt, flex_dihedral, molSymbol)
            mols.append(mol)
        return mols

    # TODO write Test
    def __delitem__(self, key):
        '''
        Special method which deletes an atom or a group of atoms from AtomicStructure.
        Atom or group of atoms to be deleted must be a molecule. I.e. one can not delete a part of molecule.

        :param key: index, slice or list of indices determining atom or atoms to be deleted from the structure.
        '''
        if isinstance(key, int):
            key = [key]
        elif isinstance(key,slice):
            key = list(range(key.start, key.stop, key.step))
        try:
            i = self._molecules.index(key)
        except ValueError:
            raise RuntimeError('Can not remove from AtomicStructure object set of atoms which is not a molecule')
        inds = list(range(len(self._chemicalSymbols)))
        inds = sorted(list(set(inds) - set(key)))
        for j in key:
            del self.atoms[j]
            del self._chemicalSymbols[j]
        del self._molecules[i]
        del self.format[i]
        del self.flex_dihedral[i]
        del self.molSymbol[i]
        for i, molecule in enumerate(self._molecules):
            self._molecules[i] = []
            for j in molecule:
                self._molecules[i].append(inds.index(j))


    def removeMolecule(self, i):
        '''
        Method which deletes a molecule.

        :param i: index of the moleculu to be deleted
        '''
        del self[self._molecules[i]]

    def merge(self, indices : list, format : list, flex_dihedral : list, molSymbol : str):
        '''
        Method which turns a group of atoms belonging current structure into a molecule.
        The molecule gets 'format' and 'flex_dihedral' information which enables a construction of ZMatrix for it.

        :param indices: List of indices of atoms in the structure to be merged into a molecule.
        :param format: List of formats of atoms to be merged into a molecule (see ZMatrix definition).
        :param flex_dihedral: List of flexible dihedral angles in the molecule (see ZMatrix definition).
        :param molSymbol: Unique name of this molecule.
        '''
        if len(indices) != len(format):
            raise RuntimeError('Dimensions of molecule parameters do not match.')
        molecules = []
        frmts = []
        flex_dihedrals = []
        molSymbols = []
        for inds, frmt, fd, symbol in ((inds, frmt, fd, symbol) for inds, frmt, fd, symbol in
                         zip(self._molecules, self.format, self.flex_dihedral, self.molSymbol) if not set(inds) <= set(indices)):
            molecules.append(inds)
            frmts.append(frmt)
            flex_dihedrals.append(fd)
            molSymbols.append(symbol)
        molecules.append(indices)
        frmts.append(format)
        flex_dihedrals.append(flex_dihedral)
        molSymbols.append(molSymbol)
        self._molecules = molecules
        self.format = frmts
        self.flex_dihedral = flex_dihedrals
        self.molSymbol = molSymbols

    def __imul__(self, m):
        '''
        Special method for supporting '*=' operator. This operator creates supercell structure of current one.

        :param m: Either int or tuple (int,int,int) describing supercell parametrs in each direction.
        Single int parameter is treated  as if tuple of three equal parameters are given.
        '''
        size = len(self.atoms)
        self.atoms.__imul__(m)
        if isinstance(m, int):
            m = (m, m, m)
        molecules = copy.copy(self._molecules)
        for i in range(1, np.product(m)):
            for j, molecule in enumerate(molecules):
                molecule = (np.asarray(molecule) + i*size).tolist()
                self._molecules.append(molecule)
                self.format.append(self.format[i])
                self.flex_dihedral.append(self.flex_dihedral[i])
                self.molSymbol.append(self.molSymbol[i])
        self._chemicalSymbols *= np.product(m)
        return self

    @property
    def composition(self):
        '''
        Property-method which returns dictionary describing composition of current structure.
        
        :return: {'symbol' : 'amount'}
        '''
        molecules = {}
        for symbol in self.molSymbol:
            if symbol in molecules.keys():
                molecules[symbol] += 1
            else:
                molecules[symbol] = 1
        return molecules

    @property
    def chemicalSymbols(self):
        return self._chemicalSymbols

    @property
    def coordinates(self):
        '''
        Property-method shortcut for calculating atomic coordinates.
        '''
        return self.atoms.get_positions()

    @property
    def scaled_coordinates(self):
        '''
        Property-method shortcut for calculating scaled atomic coordinates.
        '''
        return self.atoms.get_scaled_positions()

    @property
    def scaled_mol_coordinates(self):
        '''
        Property-method for calculating scaled coordinates of molecular centers.
        '''
        return [molecule.get_center_of_mass(scaled=True) for molecule in self.molecules]

    @property
    def lattice(self):
        '''
        Property-method shortcut for retrieving structure cell in 3x3 vector form.
        '''
        return self.atoms.get_cell().copy()

    @property
    def cell(self):
        '''
        Property-method shortcut for retrieving structure cell in 3x3 vector form.
        '''
        return self.atoms.get_cell().copy()

    @property
    def cell_lengths_and_angles(self):
        '''
        Property-method shortcut for retrieving structure cell parameters: a, b, c, alpha, beta, gama.
        '''
        return self.atoms.get_cell_lengths_and_angles()

    @property
    def volume(self):
        '''
        Property-method shortcut for calculating structure volume.
        '''
        return self.atoms.get_volume()

    @property
    def dielectricTensor(self):
        '''
        Property-method shortcut for retrieving dielectricTensor if any.
        '''
        return None if self._dielectricTensor is None else self._dielectricTensor.copy()

    # @property
    # def forces(self):
    #     return None if self.forces is None else self.forces.copy()

    @property
    def pressureTensor(self):
        '''
        Property-method shortcut for retrieving pressureTensor if any.
        '''
        return None if self._pressureTensor is None else self._pressureTensor.copy()

    @property
    def covalentRadii(self):
        '''
        Property-method for calculating covalent radii of atoms.
        :return:
        '''
        return np.fromiter((Element(atom).covalent_radius for atom in self.get_chemical_symbols()), dtype=float)

    def principleAxis(self):
        '''
        Method whSich calculates principle axes of the structure.
        Principle axes are the main axes of inertia tensor (with all atom masses set to be equal).

        :return: 3x3 array of principle axes.
        '''
        coordinates = self.coordinates - self.coordinates.mean(axis=0)
        Inertia = np.zeros((3, 3), dtype=float)  # moment of inertia tensor
        Inertia[0,0] = (coordinates[:, 1] ** 2 + coordinates[:, 2] ** 2).sum()
        Inertia[1,1] = (coordinates[:, 0] ** 2 + coordinates[:, 2] ** 2).sum()
        Inertia[2,2] = (coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2).sum()
        Inertia[0,1] = -(coordinates[:, 0] * coordinates[:, 1]).sum()
        Inertia[1,2] = -(coordinates[:, 1] * coordinates[:, 2]).sum()
        Inertia[2,0] = -(coordinates[:, 2] * coordinates[:, 0]).sum()
        Inertia[1,0] = -(coordinates[:, 0] * coordinates[:, 1]).sum()
        Inertia[2,1] = -(coordinates[:, 1] * coordinates[:, 2]).sum()
        Inertia[0,2] = -(coordinates[:, 2] * coordinates[:, 0]).sum()
        return np.linalg.eigh(Inertia)

    #TODO make it private
    def molecule_CN(self,i):
        '''
        Method roughly (very roughly!!!) estimate coordination numbers of a molecule.
        
        :param i: Molecule index.
        
        :return: Array of coordination numbers.
        '''
        molecule = self.molecules[i]
        radiu = np.array([Element(atom).covalent_radius for atom in molecule.get_chemical_symbols()])
        CN = np.fromiter((len(neighbours) for neighbours in find_pair(molecule.coordinates, radiu)), dtype = int)
        return CN

    def rotatePrinciple(self, i, axis, angle):
        assert axis < 3
        molecule = self.molecules[i]
        molecule.set_masses([1] * len(molecule))
        values, vectors = molecule.get_moments_of_inertia(vectors=True)
        ref = np.max(values)
        value = values[axis]
        vector = vectors[axis]
        if value > 0.0001:
            angle *= ref/value
            molecule.rotate(vector,angle)
        positions = self.atoms.arrays.get('positions')
        assert len(positions[self._molecules[i]]) == len(molecule.get_positions())
        positions[self._molecules[i]] = molecule.get_positions()

    def rotateFlexDiherdal(self, i, j, angle):
        assert j < len(self.flex_dihedral[i])
        molecule = self.molecules[i]
        zmatrix=coord2Zmatrix(molecule.coordinates, np.asarray(self.format[i], dtype=int))
        molecule_mod = copy.deepcopy(molecule)
        zmatrix[self.flex_dihedral[i][j],2] += angle
        molecule_mod.set_positions(zmatrix2coord(zmatrix, np.asarray(self.format[i], dtype=int)))
        if not np.array_equal(molecule.molecule_CN(0),molecule_mod.molecule_CN(0)):
            raise RuntimeError("CN mismatch")
        molecule.set_positions(molecule_mod.get_positions())
        positions = self.atoms.arrays.get('positions')
        assert len(positions[self._molecules[i]]) == len(molecule.get_positions())
        positions[self._molecules[i]] = molecule.get_positions()

    def optimizeLattice(self):
        '''
        Method which optimizes cell in case of ill formed (too prolongated) structures.
        '''
        coor = self.get_positions()
        lat = self.get_cell()
        coor, lat = optLattice(coor, lat)
        lat = AtomicStructure(cell = lat).get_cell_lengths_and_angles()
        self.set_cell(lat)
        self.set_positions(coor)

    def set_cell(self, cell, optimize = False, **kwargs):
        '''
        Method which sets up cell for the structure and calls optimizeLattice if needed.

        :param cell: 3x3 array of cell parameters to be set for the structure.
        :param optimize: boolean parameter if we should optimize lattice when setting it.
        '''
        self.atoms.set_cell(cell, **kwargs)
        if optimize:
            self.optimizeLattice()

    def __copy__(self):
        '''
        Special method which allows correctly make a copy of current structure using 'copy()' operator.
        :return: A copy of the structure with conserving type of it.
        '''
        return self.fromDICT(self.toDICT())

    def translate_scaled(self, displacement):
        '''
        Method which translates the structure on given scaled displacement.

        :param displacement: Vector of scaled distances.
        '''
        self.atoms.translate(np.dot(displacement, self.atoms.cell))

    def isGoodDistances(self, symbols : list, minDistMatrix : np.ndarray) -> bool:
        '''
        Method which checks if the structure meet minimal distance constraint provided with Minimal Distances Matrix.

        :param symbols: List of chemical symbols used in minimal distances matrix.
        :param minDistMatrix: minimal distances matrix.

        :return: True if the structure meet the constraint, false otherwise.
        '''
        if len(self.atoms) < 2:
            return True
        indices = np.fromiter((symbols.index(symbol) for symbol in self._chemicalSymbols), dtype=int)
        mDM = minDistMatrix[tuple(np.meshgrid(indices, indices))]
        actualDistances = self.get_all_distances(mic=np.any(self.atoms.get_pbc()))
        constNeighbours = np.vstack([np.eye(3), -np.eye(3)])
        for inds, molecule in zip(self._molecules, self.molecules):
            distVectorsMatrix = molecule.get_all_distances(mic = True, vector = True)
            for i, distVectorsRow in enumerate(distVectorsMatrix):
                for j, vect in enumerate(distVectorsRow):
                    vect = self.cell.scaled_positions(vect)
                    if np.all(np.abs(vect) < 1.0):
                        dists = np.linalg.norm(vect + constNeighbours, axis=1)
                        distVectorsMatrix[i,j] = self.cell.cartesian_positions(vect + constNeighbours[np.argmin(dists)])
            actualDistances[tuple(np.meshgrid(inds, inds))] = np.linalg.norm(distVectorsMatrix, axis=2)
        return np.all(actualDistances >= mDM.T)

    def isMoleculesDistinct(self) -> bool:
        '''
        Method which checks if molecules does not interpenetrate each other.
        Algorithm is as follows. For each atom distances to every other atom are calculated
        and then normalized over sum of covalent radii. The atom with shortest such distance should be in the same molecule
        as first one.
        :return true if check passed, false otherwise:
        '''
        actualMinDistances = self.get_all_distances(mic=np.any(self.atoms.get_pbc()))
        noPbcMinDistances =  self.get_all_distances(mic=False)
        normalizedMinDistances = actualMinDistances/(self.covalentRadii.reshape(-1,1) + self.covalentRadii.reshape(1,-1))
        for inds in self._molecules:
            if len(inds) > 1:
                for i in inds:
                    j = np.argsort(normalizedMinDistances[i])[1]
                    if (j not in inds) or (not np.isclose(actualMinDistances[i,j], noPbcMinDistances[i,j])):
                        return False
        return True

    def toDICT(self) -> dict:
        '''
        Method which creates dictionary representation of the structure.

        :return: Dictionary representing the structure.
        '''
        dct = super().toDICT()
        dct['symbols'] = self._chemicalSymbols
        del dct['_chemicalSymbols']
        dct['molecules'] = self._molecules
        del dct['_molecules']
        dct['molSymbols'] = self.molSymbol
        del dct['molSymbol']
        dct['molFormats'] = self.format
        del dct['format']
        dct['molFlexDihedrals'] = self.flex_dihedral
        del dct['flex_dihedral']
        dct['cell'] = self.atoms.get_cell().tolist()
        dct['positions'] = self.atoms.get_positions().tolist()
        dct['pbc'] = self.get_pbc().tolist()
        charges = self.atoms.get_initial_charges()
        if np.any(charges):
            dct['charges'] = charges.tolist()
        del dct['atoms']
        if self._dielectricTensor is not None:
            dct['dielectricTensor'] = self._dielectricTensor.tolist()
            del dct['_dielectricTensor']
        if self.forces is not None:
            dct['forces'] = self.forces.tolist()
        if self._pressureTensor is not None:
            dct['pressureTensor'] = self._pressureTensor.tolist()
            del dct['_pressureTensor']

        # TODO refactor this
        if 'bonds' in dct:
            del dct['bonds']
        if 'f' in dct:
            del dct['f']
        if 'softmodes' in dct:
            del dct['softmodes']
        if 'lastUsedModeIter' in dct:
            del dct['lastUsedModeIter']
        return dct

    @classmethod
    def fromDICT(cls, dct : dict):
        '''
        Method which reconstructs AtoimicStructure from dictionary representation.

        :param dct: Dictionary representing the structure.
        '''
        dct = copy.copy(dct)
        newStructure = cls(symbols=dct['symbols'])
        del dct['symbols']
        if 'cell' in dct:
            newStructure.set_cell(dct['cell'])
            del dct['cell']
        newStructure.set_positions(dct['positions'])
        del dct['positions']
        newStructure.set_pbc(dct['pbc'])
        del dct['pbc']
        assert len(dct['molecules']) == len(dct['molFormats']) and \
               len(dct['molecules']) == len (dct['molFlexDihedrals']) and \
               len(dct['molecules']) == len(dct['molSymbols']), \
            f'Molecule spesification missmatch:' \
            f' molecules, formats, flex_dihedrals, symbols:' \
            f' {len(dct["molecules"])}, {len(dct["molFormats"])},' \
            f' {len (dct["molFlexDihedrals"])}, {len(dct["molSymbols"])}.'
        for molecule, frmt, flex_dihedral, molSymbol in zip(dct['molecules'], dct['molFormats'], dct['molFlexDihedrals'], dct['molSymbols']):
            newStructure.merge(molecule, frmt, flex_dihedral, molSymbol)
        del dct['molecules']
        del dct['molFormats']
        del dct['molFlexDihedrals']
        del dct['molSymbols']
        if 'charges' in dct:
            newStructure.set_initial_charges(dct['charges'])
            del dct['charges']
        if 'dielectricTensor' in dct:
            newStructure._dielectricTensor = dct['dielectricTensor']
            del dct['dielectricTensor']
        if 'pressureTensor' in dct:
            newStructure._pressureTensor = np.asarray(dct['pressureTensor'])
            del dct['pressureTensor']
        if 'forces' in dct:
            newStructure.forces = np.asarray(dct['forces'])
            del dct['forces']

        newStructure.__dict__.update(dct)
        return newStructure

    @System.isBad.setter
    def isBad(self, value : bool):
        System.isBad.fset(self, value)
        if value:
            self.enthalpy = np.inf
