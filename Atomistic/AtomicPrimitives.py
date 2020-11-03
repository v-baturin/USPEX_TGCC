from copy import copy


class AtomicStructure:

    def __init__(self, atomTypes, coordinates, cell = None, edges = None, zmatrixConfig = None, **kwargs):
        assert len(atomTypes) == len(coordinates)
        self.atomTypes = copy(atomTypes)
        self.coordeinates = copy(coordinates)
        self.cell = copy(cell)
        self.edges = copy(edges)
        self.zmatrixConfig = copy(zmatrixConfig)

    def __len__(self):
        return len(self.atomTypes)

    def getAtomTypes(self):
        return copy(self.atomTypes)

    def getCortesianCoordinates(self):
        return copy(self.coordeinates)

    def getFractionalCooordinates(self):
        if self.cell is not None:
            return self.cell.cartesianToFractional(self.coordeinates)
        else:
            raise RuntimeError("Call for fractional coordinates when cell is not set up.")

    def getCenterOfMassCartesianCoordinates(self):
        pass

    def getCenterOfMassFractionalCoordinates(self):
        pass

    def getCell(self):
        return copy(self.cell)

    def getTransformedStructure(self):
        pass

    @staticmethod
    def initFormFractionalCoordinates(atomTypes, coordinates, cell, **kwargs):
        return AtomicStructure(atomTypes, cell.fractionalToCartesian(coordinates), cell, **kwargs)

    @staticmethod
    def assemble(molecules, cell, environment, **kwargs):
        atomTypes = []
        coordinates = []
        moleculesData = []
        for molecule in molecules:
            atomTypes.extend(molecule.atomTypes)
            coordinates.extend(molecule.coordinates)
            moleculesData.append({'size': len(molecule), 'cell': molecule.cell, 'edges': molecule.edges,
                                  'zmatrixConfig': molecule.zmatrixConfig})
        offsetVector = environment.calculateOffset(molecules, cell)
        structure = AtomicStructure(atomTypes, coordinates, cell, **kwargs)
        structure.translate(offsetVector)
        coordinates = structure.getCortesianCoordinates()
        atomTypes.extend(environment.getStructure().getAtomTypes())
        coordinates.extend(environment.getStructure().getCortesianCoordinates())
        return (AtomicStructure(atomTypes, coordinates, cell, **kwargs),
                AtomicDisassembler(moleculesData, environment))


class AtomicDisassembler:

    def __init__(self, moleculesData, environment):
        self.moleculesData = copy(moleculesData)
        self.environment = copy(environment)

    def disassemble(self, atomicStructure):
        atomTypesNotYet = atomicStructure.getAtomTypes()
        coordinatesNotYet = atomicStructure.getCartesianCoordinates()
        molecules = []
        for moleculeData in self.moleculesData:
            moleculeSize = moleculeData['size']
            atomTypes = atomTypesNotYet[:moleculeSize]
            del atomTypesNotYet[:moleculeSize]
            coordinates = coordinatesNotYet[:moleculeSize]
            del coordinatesNotYet[:moleculeSize]
            molecule = AtomicStructure(atomTypes, coordinates, moleculeData['cell'],
                                       edges=moleculeData['edges'], zmatrixConfig=moleculeData['zmatrixConfig'])
            molecules.append(molecule)
        assert len(atomTypesNotYet) == len(coordinatesNotYet)
        assert len(coordinatesNotYet) == len(self.environment.getStructure())
        return {'molecules': molecules, 'cell': atomicStructure.getCell(), 'environment': copy(self.environment)}


class AtomicInvariants:

    def __init__(self):
        pass

    def getFingerprint(self):
        pass

    def __eq__(self, other):
        pass
