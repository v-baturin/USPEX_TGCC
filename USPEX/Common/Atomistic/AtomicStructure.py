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

from .optLattice import optLattice
from .Element import Element
from .mol.coord2Zmatrix import coord2Zmatrix
from .mol.zmatrix2coord import zmatrix2coord
from .mol.find_pair import find_pair

THRESHOLD_LS = 0.5
THRESHOLD_HS = 1.5


class AtomicStructure(object):
    '''

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
                self._chemicalSymbols.append(s)
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

        if magmoms is not None:
            assert len(magmoms) == len(self.atoms)
            self.magmoms = magmoms
        else:
            self.magType = self.get_magnetic_type()

        super().__init__()

    def __getattr__(self, item):
        if item == '__setstate__':
            raise AttributeError
        if hasattr(self.atoms, item):
            return getattr(self.atoms, item)
        else:
            raise AttributeError

    def __len__(self):
        return len(self.atoms)

    def extend(self, ext):
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
        res = copy.copy(self)
        res.extend(other)
        return res

    def __getitem__(self, item):
        return self.atoms.__getitem__(item)

    @property
    def molecules(self):
        mols = []
        for inds, frmt, flex_dihedral, molSymbol in zip(self._molecules, self.format, self.flex_dihedral, self.molSymbol):
            mol = AtomicStructure(symbols=self[inds], cell=self.get_cell())
            if len(mol) > 1:
                mol.merge(list(range(len(mol))), frmt, flex_dihedral, molSymbol)
            mols.append(mol)
        return mols

    # TODO write Test
    def __delitem__(self, key):
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
        del self[self._molecules[i]]

    def merge(self, indices : list, format : list, flex_dihedral : list, molSymbol : str):
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
        :return: 
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
        return self.atoms.get_positions()

    @property
    def scaled_coordinates(self):
        return self.atoms.get_scaled_positions()

    @property
    def scaled_mol_coordinates(self):
        return [molecule.get_center_of_mass(scaled=True) for molecule in self.molecules]

    @property
    def lattice(self):
        return self.get_cell().copy()

    @property
    def cell(self):
        return self.atoms.get_cell()

    @property
    def cell_lengths_and_angles(self):
        return self.atoms.get_cell_lengths_and_angles()

    @property
    def atomTypes(self):
        return np.unique(self._chemicalSymbols)

    @property
    def atom_type_seq(self):
        tmp = []
        res = []
        count = -1
        for symbol in self._chemicalSymbols:
            if symbol not in tmp:
                tmp.append(symbol)
                count += 1
            res.append(count)
        return res

    @property
    def volume(self):
        return self.atoms.get_volume()

    @property
    def dielectricTensor(self):
        return None if self._dielectricTensor is None else self._dielectricTensor.copy()

    # @property
    # def forces(self):
    #     return None if self.forces is None else self.forces.copy()

    @property
    def pressureTensor(self):
        return None if self._pressureTensor is None else self._pressureTensor.copy()

    @property
    def magmoms(self):
        return self.atoms.get_initial_magnetic_moments()

    @magmoms.setter
    def magmoms(self, spins):
        self.atoms.set_initial_magnetic_moments(spins)
        self.magType = self.get_magnetic_type()

    def round_magnetic_moments(self):
        new_magmoms = np.zeros(len(self.magmoms))
        for i, magmom in enumerate(self.magmoms):
            if abs(magmom) > THRESHOLD_HS:
                new_magmoms[i] = 4 * np.sign(magmom)
            elif THRESHOLD_LS <= abs(magmom) <= THRESHOLD_HS:
                new_magmoms[i] = np.sign(magmom)
        self.magmoms = new_magmoms

    def get_magnetic_type(self):
        if all(abs(self.magmoms) < 0.5):
            return 'NM'
        else:
            nonzero = self.magmoms[abs(self.magmoms) >= THRESHOLD_LS]
            spinsup = nonzero[np.sign(nonzero) > 0]
            spinsHS = nonzero[abs(nonzero) > THRESHOLD_HS]
            if 0.25 < len(spinsup) / len(nonzero) < 0.75:  # AFM
                if len(spinsHS) < (1 / 3) * len(nonzero):
                    return 'AFM-LS'
                elif len(spinsHS) > (2 / 3) * len(nonzero):
                    return 'AFM-HS'
                else:
                    return 'AFM-HSLS'
            else:  # FM
                if len(spinsHS) < (1 / 3) * len(nonzero):
                    return 'FM-LS'
                elif len(spinsHS) > (2 / 3) * len(nonzero):
                    return 'FM-HS'
                else:
                    return 'FM-HSLS'

    def principleAxis(self):
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

    def molecule_CN(self,i):
        molecule = self.molecules[i]
        radiu = np.array([Element(atom).covalent_radius for atom in molecule.get_chemical_symbols()])
        CN = np.fromiter((len(neighbours) for neighbours in find_pair(molecule.coordinates, radiu)), dtype = int)
        return CN

    def RotInertia(self,i):
        molecule = self.molecules[i]
        molecule.set_masses([1] * len(molecule))
        values, vectors = molecule.get_moments_of_inertia(vectors=True)
        #b(1,1) =0 for linear chain molecules, thus two rotational variables
        if values[0] < 0.0001:
            ref = values[1]
        else:
            ref = values[0]

        for value, vector in zip(values, vectors):
            if value > 0.0001:
                angle = ( np.pi/2*np.random.random_sample() - np.pi/4)*ref/value
                molecule.rotate(vector,angle)

        molecule.translate(np.random.random_sample(3)-0.5)

        zmatrix=coord2Zmatrix(molecule.coordinates, np.asarray(self.format[i], dtype=int))

        if len(self.flex_dihedral[i]) > 0:
            goodRot = 0

            molecule_mod = copy.deepcopy(molecule)
            while not goodRot:
                for j in self.flex_dihedral[i]:
                    zmatrix[j,2] = zmatrix[j,2]# + ( np.pi*np.random.random_sample() - np.pi/2)
                molecule_mod.set_positions(zmatrix2coord(zmatrix, np.asarray(self.format[i], dtype=int)))
                if np.array_equal(molecule.molecule_CN(0),molecule_mod.molecule_CN(0)):
                    goodRot = 1
            molecule.set_positions(molecule_mod.get_positions())
        positions = self.atoms.arrays.get('positions')
        assert len(positions[self._molecules[i]]) == len(molecule.get_positions())
        positions[self._molecules[i]] = molecule.get_positions()

    def optimizeLattice(self):
        coor = self.get_positions()
        lat = self.get_cell()
        coor, lat = optLattice(coor, lat)
        lat = AtomicStructure(cell = lat).get_cell_lengths_and_angles()
        self.set_cell(lat)
        self.set_positions(coor)

    def set_cell(self, cell, optimize = False, **kwargs):
        self.atoms.set_cell(cell, **kwargs)
        if optimize:
            self.optimizeLattice()

    def __copy__(self):
        dct = self.toDICT()
        if 'ID' in dct:
            del dct['ID']
        return self.fromDICT(dct)

    def translate_frac(self, displacement : np.array):
        assert len(displacement) == 3
        self.atoms.translate(np.dot(displacement, self.atoms.cell))

    def isGoodDistances(self, symbols : list, minDistMatrix : np.ndarray) -> bool:
        if len(self.atoms) < 2:
            return True
        indices = np.fromiter((symbols.index(symbol) for symbol in self._chemicalSymbols), dtype=int)
        mDM = minDistMatrix[np.meshgrid(indices, indices)]
        for inds, molecule in zip(self._molecules, self.molecules):
            if len(inds) == 1:
                mDM[inds[0], inds[0]] = 0
            else:
                minDist = mDM[np.meshgrid(inds, inds)]
                molDist = molecule.get_all_distances(mic = False) - 0.02
                mDM[np.meshgrid(inds, inds)] = np.minimum(minDist, molDist)
        return np.all(self.get_all_distances(mic=np.any(self.atoms.get_pbc())) >= mDM.T)

    def toJSON(self) -> str:
        return json.dumps(self.toDICT())

    def toDICT(self) -> dict:
        dct = copy.copy(self.__dict__)
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
        dct['magmoms'] = self.magmoms.tolist()
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

    @staticmethod
    def fromJSON(repr : str):
        dct = json.loads(repr)
        return AtomicStructure.fromDICT(dct)

    @classmethod
    def fromDICT(cls, dct : dict):
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
        for molecule, frmt, flex_dihedral, molSymbol in zip(dct['molecules'], dct['molFormats'], dct['molFlexDihedrals'], dct['molSymbols']):
            newStructure.merge(molecule, frmt, flex_dihedral, molSymbol)
        del dct['molecules']
        del dct['molFormats']
        del dct['molFlexDihedrals']
        del dct['molSymbols']
        if 'charges' in dct:
            newStructure.set_initial_charges(dct['charges'])
            del dct['charges']
        if 'magmoms' in dct:
            newStructure.magmoms = dct['magmoms']
            del dct['magmoms']
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