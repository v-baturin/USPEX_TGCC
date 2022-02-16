"""
USPEX.Atomistic.EnvironmentUtility
=====================
"""

from ase.io import read
import numpy as np

class Substrate:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """
    
    _DEFAULT_SUBSTRATE_SHIFT = 2.0

    def __init__(self, structure, variableThickness = None, offsetVector = None):
        """
        :param structure: atomic structure associated with this environment.
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
            If not provided such vector will be calculated on demand.
        """
        self._structure = structure
        self._thickness = variableThickness
        self._offsetVector = offsetVector
        antiPBC = self._structure.getCell().getAntiPBC()
        assert sum(antiPBC) == 1
        self._ind = np.flatnonzero(antiPBC)[0]
        coordinates = self._structure.getCartesianCoordinates()[:, self._ind]
        upperBound = coordinates.max() - self._thickness if variableThickness is not None else coordinates.min()
        self._indices = np.flatnonzero(coordinates < upperBound)

    def getThickness(self):
        return self._thickness

    def calculateOffset(self, molecules, cell):
        """
        Calculate or retrieve vector to be added to each molecule when assemble whole structure.
        If such vector is not predefined for this environment it will be calculated basing on minimal atomic coordinates
        in nonperiodic direction.

        :param molecules: list of molecules for which the offset is being calculated.
        :param cell: TODO

        :return: offset vector.
        """
        if self._offsetVector is not None:
            offsetVector = self._offsetVector
        else:
            cell = self._structure.getCell()
            axis = cell.getCellVectors()[self._ind]
            envCoordinates = cell.cartesianToFractional(self._structure.getCartesianCoordinates())[:, self._ind]
            coordinates = np.vstack([cell.cartesianToFractional(molecule.getCartesianCoordinates())
                                     for molecule in molecules])[:, self._ind]
            offsetVector = axis * (self._DEFAULT_SUBSTRATE_SHIFT / np.linalg.norm(axis)
                                   + envCoordinates.max() - coordinates.min())
        return offsetVector
        
    def getStructure(self):
        """
        Retrieve atomic structure associated with environment.

        :return: atomic structure.
        """
        return self._structure

    def getFixedIndices(self):
        return self._indices

class EnvironmentUtility:
    """
    Class reprenting utility which generates possible environmemnts for calculation.
    """
    structureRepresentation = None

    @classmethod
    def setRepresentation(cls, representation):
        cls.structureRepresentation = representation

    def __init__(self, type = None, files = None, pbcs = None, **kwargs):
        """

        :param files: files with structures for possile environments.
        :param pbc: periodic boundary conditions of environment structures.

        """
        assert type is None or type == 'Substrate'
        assert (files is None) == (pbcs is None)
        self._kwargs = kwargs
        self._files = files if files is not None else []
        self._pbcs = pbcs if pbcs is not None else []
        self._structures = []
        for file, pbc in zip(self._files, self._pbcs):
            structure = self.structureRepresentation.readAtomicStructureRaw(file, pbc)
            self._structures.append(structure)

    def putEnvironment(self, system):
        """
        Put environment in dictionary representing system.

        :param system: system dictionary.

        """
        if self._structures:
            structure = np.random.choice(self._structures)
            structure = structure.makeSupercell(np.round(structure.getCell().decomposeCell(system['cell'])))
            system['environment'] = Substrate(structure, **self._kwargs)
