"""
USPEX.Atomistic.EnvironmentUtility
==================================
"""

import numpy as np
from copy import copy


class Substrate:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """
    
    _DEFAULT_SUBSTRATE_SHIFT = 2.0

    def __init__(self, structure, bufferThickness = None, offsetVector = None): # TODO: add tiling (maybe)
        """
        :param structure: atomic structure associated with this environment.
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
            If not provided such vector will be calculated on demand.
        """
        self._structure = structure
        self._thickness = bufferThickness  # comes from input
        self._offsetVector = offsetVector
        antiPBC = self._structure.getCell().getAntiPBC()
        assert sum(antiPBC) == 1
        self._ind = np.flatnonzero(antiPBC)[0]
        coordinates = self._structure.getCartesianCoordinates()[:, self._ind]
        upperBound = coordinates.max() - self._thickness if self._thickness is not None else coordinates.min()
        self._indices = np.flatnonzero(coordinates < upperBound)

    def getThickness(self):
        """
        Get thickness of the substrate.
        """
        return self._thickness

    def calculateOffset(self, molecules, syscell=None):
        """
        Calculate or retrieve vector to be added to each molecule when assemble whole structure.
        If such vector is not predefined for this environment it will be calculated basing on minimal atomic coordinates
        in nonperiodic direction.

        :param molecules: list of molecules for which the offset is being calculated.
        :param cell: TODO

        :return: offset vector.
        """
        if self._offsetVector is not None:
            offsetVector = np.asarray(self._offsetVector, dtype=float)
        else:
            cell = self._structure.getCell()
            frac_coords = type(self._structure).assemble(molecules, cell)[0].getFractionalCoordinates()
            sysPBC = syscell.getPBC()
            offsetVector = np.zeros(3)
            for idx in range(3):
                curr_axis = cell.getCellVectors()[idx]
                if idx == self._ind:
                    fracEnvCoordinates = cell.cartesianToFractional(self._structure.getCartesianCoordinates())
                    offsetVector += curr_axis * (self._DEFAULT_SUBSTRATE_SHIFT / np.linalg.norm(curr_axis)
                                   + fracEnvCoordinates[:, idx].max() - frac_coords[:, idx].min())
                elif not sysPBC[idx]:
                    offsetVector += curr_axis * (0.5 - 0.5 * (frac_coords[:, idx].min() + frac_coords[:, idx].max()))
        return offsetVector

    def getUpdatedEnvironment(self, atomTypes, coordinates, cell, envStructure):
        return Substrate(envStructure, self._thickness, np.min(coordinates, axis=0))
        
    def getStructure(self):
        """
        Retrieve atomic structure associated with environment.

        :return: atomic structure.
        """
        return self._structure

    def getFixedIndices(self):
        """
        Get indices of atoms in substrate positions of which are fixed.
        """
        return self._indices


class Bulk:

    def __init__(self, structure, isFixed: bool):
        self._structure = structure
        self.isFixed = isFixed
        if self.isFixed:
            self._indices = np.arange(len(structure))
        else:
            self._indices = np.array([], dtype=int)

    def calculateOffset(self, molecules, syscell=None):
        return np.array([0.0, 0.0, 0.0])

    def getUpdatedEnvironment(self, atomTypes, coordinates, cell, envStructure):
        return Bulk(envStructure, self.isFixed)

    def getStructure(self):
        """
        Retrieve atomic structure associated with environment.

        :return: atomic structure.
        """
        return self._structure

    def getFixedIndices(self):
        """
        Get indices of atoms in substrate positions of which are fixed.
        """
        return self._indices


class EnvironmentUtility:
    """
    Class representing utility which generates possible environmemnts for calculation.
    """
    structureRepresentation = None

    @classmethod
    def setRepresentation(cls, representation):
        cls.structureRepresentation = representation

    def __init__(self, environments: list = None):
        """

        :param files: files with structures for possile environments.
        :param pbc: periodic boundary conditions of environment structures.

        """
        self._environments = []
        if environments is not None:
            for environment in environments:
                file = environment.pop('file')
                pbc = environment.pop('pbc')
                environment['structure'] = self.structureRepresentation.readAtomicStructureRaw(file, pbc)
                self._environments.append(environment)

    def hasEnvironment(self):
        return len(self._environments) > 0

    def putEnvironment(self, system, environment=None):
        """
        Put environment in dictionary representing system.

        :param system: system dictionary.

        """
        if environment is not None:
            system['environment'] = environment
        elif self._environments:
            environment = copy(np.random.choice(self._environments))
            name = environment.pop('name')
            envType = environment.pop('type')
            structure = environment.pop('structure')
            environment['structure'] = structure.makeSupercell(np.round(structure.getCell().decomposeCell(system['cell'])))
            if envType == 'substrate':
                system['environment'] = Substrate(**environment)
            elif envType == 'bulk':
                system['environment'] = Bulk(**environment)
            else:
                raise ValueError(f"Unknown environment type {envType}.")
