"""
USPEX.Atomistic.SimpleMoleculeUtility
=====================================
"""

import numpy as np
from collections import Counter

from .Transformation import Transformation


DENSITY_CONST = 1.660539


class SimpleMoleculeUtility(object):
    """
    Utility providing methods for work with simple molecules.
    """

    structureType = None
    atomType = None

    @classmethod
    def registerTypes(cls, structureType, atomType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        """
        cls.structureType = structureType
        cls.atomType = atomType

    def __init__(self, molecules: dict = None, doCenterMolecule=False):
        """
        :param molecules: {<name>: <definition>} dictionary of molecule definitions.

        """
        self.isTrueMolecular = bool(molecules)
        self.molecules = {el.short_name: self.structureType([el], [[0., 0., 0.]]) for el in self.atomType.all_elements()}
        if molecules is not None:
            for symbol, molecule in molecules.items():
                if doCenterMolecule:
                    offset = Transformation.fromRotVector([0., 0., 0.], -molecule.getCenterOfMassCartesianCoordinates())
                    molecule = offset.transform(molecule)
                self.molecules[symbol] = molecule
        self.formulaToTypeMap = {molecule.getFormula() : molSymbol for molSymbol, molecule in self.molecules.items()}
        # TODO: what if we have two molecules with same formula?


    def populateStructure(self, cell, operations):
        """
        Creates list list of molecules by placing corresponding molecule in place specified by map *coordinates*
        in orientation specified by map operations.

        :param cell: unit cell.
        :param coordinates: map {<molecule_symbols> : <center_coordinates>}
        :param operations: map {<molecule_symbols> : <orientation>}

        :return: list of molecules.
        """
        molecules = []
        optimizedCell = cell.getOptimizedCell()
        for symbol, speciesOperations in operations.items():
            molecule = self.molecules[symbol]
            for variants in speciesOperations:
                if len(molecule) > 1:
                    molecule = Transformation.fromRotVector(Transformation.randomRotVector(), [0., 0., 0.]).transform(molecule)
                for operation in variants[np.random.randint(len(variants))]:
                    transformation = cell.fractionalToCartesianOperator(operation)
                    matrix, position = (transformation.rotMatrix, transformation.transVec)
                    position = optimizedCell.getWrapedCartesianCoordinates(position)
                    transformation = Transformation(matrix, position)
                    molecules.append(transformation.transform(molecule))
        return {'molecules': molecules, 'cell': optimizedCell}

    def determineMoleculeType(self, molecule):
        """
        Assuming every type of molecules defined in calculation has unique formula,
        determines molecule symbol of given molecule.

        :param molecule: atomic structure representing molecule.

        :return: molecule symbol.
        """
        return self.formulaToTypeMap[molecule.getFormula()]

    def moleculeTypes(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated or retrieve list of types of molecules of a system.
        """
        if 'simpleMoleculeUtility.moleculeTypes' not in system:
            moleculeTypes = [self.determineMoleculeType(molecule) for molecule in system['molecules']]
            system.setProperty('simpleMoleculeUtility.moleculeTypes', moleculeTypes)
        return system['simpleMoleculeUtility.moleculeTypes']

    def composition(self, system):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated or retrieve molecular composition of a system.
        """
        if 'simpleMoleculeUtility.composition' not in system:
            composition = Counter(dict(zip(*np.unique(self.moleculeTypes(system), return_counts=True))))
            system.setProperty('simpleMoleculeUtility.composition', composition)
        return system['simpleMoleculeUtility.composition']

    def density(self, system):
        cell = system['cell']
        if cell.dim == 3:
            mass = sum(e.mass*v for e, v in self.getElementalComposition(self.composition(system)).items())
            return mass/cell.getVolume()*DENSITY_CONST
        else:
            return None

    def getElementalComposition(self, composition):
        """
        For given molecular composition {<molecule_symbol> : <amount>} calculates elemental composition {<element> : <amount>}.

        :param composition: molecular composition.

        :return: elemental composition.
        """
        comp = Counter()
        for symbol, amount in composition.items():
            for el, value in self.molecules[symbol].getComposition().items():
                comp[el] += value*amount
        return comp

    def checkMinDistances(self, entry, minDistMatrix):
        """
        Calculates minimal distances between atoms excluding intramolecular distances.

        :param molecules: list of molecules.
        :param cell: unit cell.

        :return: matrix of distances between atoms.
        """

        # positions = self.atoms.get_positions()
        # i_init, j_init, vect = primitive_neighbor_list(quantities='ijD', pbc=self.atoms.pbc, cell=self.atoms.get_cell(complete=True),
        #                                          positions=positions, cutoff=minDistMatrix,
        #                                          numbers=self.atoms.numbers, use_scaled_positions=False)
        #
        # # whether system is molecular or not
        # # Check whether all pairs of atoms, which are closer than minDistMatrix and are related to the same molecule
        # for i, j, v in zip(i_init, j_init, vect):
        #     inMolecule = False
        #     for mol in self._molecules:
        #         if i in mol and j in mol and np.allclose(v, positions[j] - positions[i]):
        #             inMolecule = True
        #             break
        #     if not inMolecule: return False
        # return True

        molecules = entry['molecules']
        cell = entry['cell']
        structure = entry.getAtomicStructure()
        disassembler = entry['disassembler']
        actualDistances = structure.getAllDistances()
        eye = np.eye(3)[np.nonzero(cell.getPBC())]
        constNeighbours = np.vstack([eye, -eye])
        for inds, molecule in zip(disassembler.indices, molecules):
            distVectorsMatrix = molecule.getAllPairVectors()
            for i, distVectorsRow in enumerate(distVectorsMatrix):
                for j, vect in enumerate(distVectorsRow):
                    vect = cell.cartesianToFractional(vect)
                    if np.all(np.abs(vect) < 1.0):
                        dists = np.linalg.norm(vect + constNeighbours, axis=1)
                        distVectorsMatrix[i,j] = cell.fractionalToCartesian(vect + constNeighbours[np.argmin(dists)])
            actualDistances[tuple(np.meshgrid(inds, inds))] = np.linalg.norm(distVectorsMatrix, axis=2)
        for inds in disassembler.envIndices:
            actualDistances[tuple(np.meshgrid(inds, inds))] = minDistMatrix[
                tuple(np.meshgrid(inds, inds))]

        return np.all(actualDistances >= minDistMatrix)

    @staticmethod
    def rotationClearance(inertiaValues):
        """
        Calculates relative mobility of structure around axes with given inertia values.

        :param inertiaValues: inertia values of axes.

        :return: relative clearances around axes.
        """
        minValue = np.min(inertiaValues)
        if np.isclose(minValue, 0):
            clearance = np.zeros(inertiaValues.shape)
            clearance[np.argmin(inertiaValues)] = np.pi
        else:
            clearance = np.pi*minValue/inertiaValues
        return clearance

    @classmethod
    def rotateFlexDiherdal(cls, molecule, i: int, angle: float):
        """
        Rotate the *i* th flexible dihedral angle of molecule by the angle *angle* .

        :param i: index of flexible dihedral angle to rotate.
        :param angle: angle by which the dihedral should be rotated.
        """
        assert i < len(molecule.zmatrixConfig.flex_dihedral)
        zmatrix = cls.coordToZmatrix(molecule.getCartesianCoordinates, np.asarray(molecule.zmatrixConfig.format, dtype=int))
        zmatrix[molecule.zmatrixConfig.flex_dihedral[i], 2] += angle
        molecule_mod = type(molecule)(molecule.getAtomTypes(),
                                      cls.zmatrixToCoord(zmatrix, np.asarray(molecule.zmatrixConfig.format, dtype=int)),
                                      cell = molecule.getCell())
        if np.array_equal(cls.molecule_CN(molecule_mod), cls.molecule_CN(molecule)):
            return molecule_mod
        else:
            return molecule

    @classmethod
    def molecule_CN(cls, molecule):
        """
        Method which roughly (very roughly!!!) estimates the coordination numbers of a molecule.

        :param molecule: atomic structure representing molecule.

        :return: array of coordination numbers.
        """
        radiu = np.array([atom.covalent_radius for atom in molecule.getAtomTypes()])
        CN = np.fromiter((len(neighbours) for neighbours in _find_pair(molecule.getCartesianCoordinates(), radiu)), dtype=int)
        return CN

    @staticmethod
    def zmatrixToCoord(zmatrix, fmt):
        """
        Function that transforms Z-matrix to XYZ coordinates. Remember that the Z-matrix of a molecule is defined
        in spherical coordinates, so we need a lot of transformations from (r, theta, phi) to (x, y, z).

        :type zmatrix: numpy array
        :param zmatrix:
            Z-matrix of a molecule.
        :type fmt: numpy array
        :param fmt:
            for each atom in the molecule are listed the indices of three other atoms,
            with respect to which the parameters of the Z-matrix are calculated.

        :rtype: numpy array
        :return:
            XYZ coordinates of atoms in the molecule.
        """
        N_atom = len(zmatrix)
        coords = np.zeros((N_atom, 3))
        origin = zmatrix[0, :]
        if N_atom > 1:
            coords[1, 2] = zmatrix[1, 0]*np.cos(zmatrix[1, 1])
            coords[1, 0] = zmatrix[1, 0]*np.sin(zmatrix[1, 1])*np.cos(zmatrix[1, 2])
            coords[1, 1] = zmatrix[1, 0]*np.sin(zmatrix[1, 1])*np.sin(zmatrix[1, 2])
            if N_atom > 2:
                for i in range(2, N_atom):
                    if i == 2:
                        ref = coords[fmt[2, :2] - 1, :]
                    else:
                        ref = coords[fmt[i, :] - 1, :]
                    coords[i, :] = _GetXYZ(ref, zmatrix[i, :])
        coords += origin
        return coords

    @staticmethod
    def coordToZmatrix(coords, fmt):
        """
        Function that transforms XYZ to Z-matrix coordinates. Remember that the Z-matrix of a molecule is defined
        in spherical coordinates, so we need a lot of transformations from (x, y, z) to (r, theta, phi).

        :type coords: numpy array
        :param coords:
            XYZ coordinates.
        :type fmt: numpy array
        :param fmt:
            for each atom in the molecule are listed the indices of three other atoms,
            with respect to which the parameters of the Z-matrix are calculated.

        :rtype: numpy array
        :return:
            Z-matrix of the molecule.
        """
        fmt = np.copy(fmt)
        coords = np.copy(coords)
        coords = np.real(coords)

        Zmatrix = np.copy(coords)  # 1st atom always = coords
        N_atom = coords.shape[0]

        if N_atom > 1:
            coords -= coords[0, :]
            # 2nd atom, define it in spherical coordinates
            Zmatrix[1, 0] = np.real(np.linalg.norm(coords[1, :]))
            if coords[1, 2] == 0:
                Zmatrix[1, 1] = np.pi * 0.5
            else:
                Zmatrix[1, 1] = np.arccos(coords[1, 2] / Zmatrix[1, 0])

            if coords[1, 1] == 0:
                Zmatrix[1, 2] = 0
            else:
                Zmatrix[1, 2] = np.arctan2(coords[1, 1], coords[1, 0])

            for ind in range(2, N_atom):
                a1 = coords[ind, :]
                a2 = coords[fmt[ind, 0] - 1, :]  # there and below: python indexing from 0
                a3 = coords[fmt[ind, 1] - 1, :]
                Zmatrix[ind, 0] = np.real(np.linalg.norm(a2 - a1))
                Zmatrix[ind, 1] = _GetAngle(a1, a2, a3)
                if ind == 2:  # the dihedral angle between 1-2-3 and XY plane
                    a4 = a3 + np.array([1.0, 0.0, 0.0])
                    # Zmatrix(ind, 3) = -1*GetDihedral(a1, a2, a3, a4);
                else:
                    a4 = coords[fmt[ind, 2] - 1, :]
                    # Zmatrix(ind, 3) = -1*GetDihedral(a1, a2, a3, a4);

                Zmatrix[ind, 2] = _GetDihedral(a1, a2, a3, a4)

        Zmatrix = np.real(Zmatrix)
        return Zmatrix

def _find_pair(coor, radii):
    """
    This function checks all the atom pairs and constructs the neighbor list.
    The bond length is estimated by the covalent radii of the atoms.

    :type coor: numpy array
    :param coor: Nx3 array of atomic coordinates.
    :type radii: numpy array
    :param radii: Nx1 array of atomic radii.

    :rtype: list of list of int
    :return: list with the indices of neighboring atoms for each atom.
    """
    n_atom = len(radii)
    # maximum 6 coordination, 7 gives the coordination number
    # pair = np.zeros((n_atom, N_max), dtype=int)
    pair = [[] for x in radii]

    for i in range(n_atom):
        for j in range(i + 1, n_atom):
            if np.linalg.norm(coor[i] - coor[j]) < 1.2 * (radii[i] + radii[j]):
                pair[i].append(j)
                pair[j].append(i)

    # we assume there is no isolated atom
    if n_atom > 1:
        for i in range(n_atom):
            if len(pair[i]) == 0:
                print('atom_{} is not connected to any other atom'.format(i))
                print('Please check your MOL file again. Serious WARNING.... ')

    return pair

def _GetAngle(a1, a2, a3):
    """
    Returns the angle between three atoms from their respective coordinates.

    :type a1: numpy array
    :param a1: 1x3 array with the coordinates of the first atom.
    :type a2: numpy array
    :param a2: 1x3 array with the coordinates of the second atom.
    :type a3: numpy array
    :param a3: 1x3 array with the coordinates of the third atom.

    :rtype: float
    :return: angle in radians between the three atoms.
    """
    v1 = a1 - a2
    v2 = a3 - a2
    angle = np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2))
    return angle

def _GetDihedral(a1, a2, a3, a4):
    """
    Returns the dihedral angle between four atoms from their respective coordinates.

    :type a1: numpy array
    :param a1: 1x3 array with the coordinates of the first atom.
    :type a2: numpy array
    :param a2: 1x3 array with the coordinates of the second atom.
    :type a3: numpy array
    :param a3: 1x3 array with the coordinates of the third atom.
    :type a4: numpy array
    :param a4: 1x3 array with the coordinates of the fourth atom.

    :rtype: float
    :return: dihedral angle in radians between the four atoms.
    """
    p = a2 - a1
    q = a3 - a2
    r = a4 - a3
    n1 = np.cross(p, q)
    n2 = np.cross(q, r)
    torsion = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    center = (a1 + a2 + a3) / 3.0
    if np.dot(n1, a4 - center) < 0:
        torsion *= -1

    return torsion

def _GetXYZ(ref, zmatrix):
    """
    Get the XYZ coordinates of the current atom from its Z-matrix coordinates
    and the XYZ coordinates of the reference atoms.

    :type ref: numpy array
    :param ref: XYZ coordinates of the reference atoms.
    :type zmatrix: numpy array
    :param zmatrix: Z-matrix coordinates of the current atom.

    :rtype: numpy array
    :return: XYZ coordinates of the current atom.
    """
    r = zmatrix[0]
    theta = zmatrix[1]
    phi = -zmatrix[2]

    coor = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
    u1 = ref[1, :] - ref[0, :]
    if len(ref) == 2:
        u2 = np.array([1, 0, 0])
    else:
        u2 = ref[2, :] - ref[1, :]

    z = u1/np.linalg.norm(u1)
    y = np.cross(u1, u2)
    y = y/np.linalg.norm(y)
    x = np.cross(y, z)
    x = x/np.linalg.norm(x)

    # coor = coor / (np.stack([x, y, z]).T)
    coor = np.linalg.lstsq(np.stack([x, y, z]), coor)[0]
    return coor + ref[0, :]
