import logging
logger = logging.getLogger(__name__)

import signal
from time import time
import numpy as np

from pyxtal.crystal import Lattice
from pyxtal import pyxtal
from pyxtal.molecular_crystal import molecular_crystal


MAX_PYXTAL_TIME = 30
MAX_RANDOM_TIME = 300
MAX_PYXTAL_ATTEMPTS = 10000
LOCAL_VACUUM = 0.2

class RandSymPyXtal:
    def __init__(self, utilities, symmetries=None, factor=1.1):
        self.atomistic = utilities.atomistic
        self.cellUtility = utilities.cellUtility
        self.environmentUtility = utilities.environmentUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        # if self.simpleMoleculeUtility.isTrueMolecular:
        #     raise RuntimeError("RandSymPyXtal does not currently work in molecular regime.")
        if isinstance(symmetries, int):
            self.nsym = [symmetries]
        elif isinstance(symmetries, str):
            self.nsym = list(parseIntSet(symmetries))
        elif symmetries is None:
            dim = self.cellUtility.getDim()
            if dim == 3:
                self.nsym = list(range(1, 231))
            elif dim == 2:
                self.nsym = list(range(1, 81))
            elif dim == 1:
                self.nsym = list(range(1, 76))
            elif dim == 0:
                self.nsym = list(range(1, 57))
            else:
                raise ValueError(f"Wrong dim {dim}.")
        else:
            self.nsym = symmetries
        self.factor = factor
        signal.signal(signal.SIGALRM, signal_handler)

    def __setstate__(self, state):
        # Set up signal handler after unpickling
        self.__dict__.update(state)
        signal.signal(signal.SIGALRM, signal_handler)

    def __call__(self, offspringFactory=None):
        composition = self.compositionSpace.randomComposition()

        symbols = list(composition.keys())
        numIons = list(composition.values())
        estimatedVolume = self.cellUtility.getCellVolume()
        if estimatedVolume is None:
            elementalComposition = self.simpleMoleculeUtility.getElementalComposition(composition)
            estimatedVolume = self.bondUtility.volumeEstimator.calcCompositionVolume(elementalComposition,
                                                                                      self.conditions.externalPressure)

        if np.sum(numIons) == 0:
            raise RuntimeError("Structure with no atoms requested. Skip.")

        # PyXtal cannot work with numIons == 0, so remove the corresponding elements
        for i in reversed(range(len(symbols))):
            if numIons[i] == 0:
                symbols.pop(i)
                numIons.pop(i)

        dim = self.cellUtility.getDim()
        startTime = time()
        failCounter = 0
        while True:
            envAssembler = np.random.choice(self.environmentUtility.environments) if self.environmentUtility.environments\
                else None
            envCell = envAssembler.getCell() if envAssembler is not None else None

            endTime = time()
            failedTime = endTime - startTime
            if failCounter > MAX_PYXTAL_ATTEMPTS or failedTime > MAX_RANDOM_TIME:
                raise RuntimeError("RandSymPyXtal failed.")

            nsym, = np.random.choice(self.nsym, 1)
            logger.debug(f"Trying {nsym} symmetry")

            #randcell is an auxiliary cell object to get required info from
            randcell = self.cellUtility.getRandomCell(estimatedVolume, sum(numIons), baseCell=envCell)
            if dim == 3 or dim == 0:
                lat = None
            elif dim == 2:
                LayerThickness = self.cellUtility.getThickness()
                LayerArea = randcell.getArea()
                lat = generate2Dcell(nsym, LayerThickness, LayerArea)
            elif dim == 1:
                CylinderRadius = self.cellUtility.getThickness() / 2.0
                CylinderLength = randcell.getLength()
                lat = generate1Dcell(nsym, CylinderRadius, CylinderLength)
            else:
                raise ValueError(f"Wrong dim {dim}.")

            if not self.simpleMoleculeUtility.isTrueMolecular:
                structurePyxtal = pyxtal()
                signal.alarm(MAX_PYXTAL_TIME)
                try:
                    structurePyxtal.from_random(dim, nsym, symbols, numIons, lattice=lat)
                except Exception as e:
                    signal.alarm(0)
                    logger.debug(e)
                    continue
                signal.alarm(0)

                if structurePyxtal.valid:
                    tmp_cell, operations = convertStruc(structurePyxtal, randcell.getPBC(), symbols, LOCAL_VACUUM)
                    cell = self.cellUtility.adjustCell(tmp_cell, estimatedVolume, sum(numIons), baseCell=envCell)
                    operations = dict(zip(symbols, operations))
                    offspring = offspringFactory(**self.simpleMoleculeUtility.populateStructure(cell, operations))
                    molecules = offspring.getProperty('molecules', extension='atomistic')
                    cell = offspring.getProperty('cell', extension='atomistic')
                    if envAssembler is not None:
                        offspring.setProperty('environments',
                                              envAssembler.assemble(molecules, cell),
                                              extension='atomistic')
                    structure = offspring.getProperty('structure', extension='atomistic')
                    minDistMatrix = self.bondUtility.getDistances(
                        structure.getAtomTypes(), self.conditions.externalPressure)
                    if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix):
                        self.conditions.putConditions(offspring)
                        if self.bondUtility.isConnected(structure):
                            return offspring,
            else:
                molecules, lattice, volume = self.molecularCrystal(dim, nsym, symbols, numIons, lattice=randcell.getCellVectors())
                cell = self.cellUtility.adjustCell(lattice, volume, sum(numIons), baseCell=envCell)
                offspring = offspringFactory(**{'atomistic.molecules': molecules, 'atomistic.cell': cell})
                molecules = offspring.getProperty('molecules', extension='atomistic')
                cell = offspring.getProperty('cell', extension='atomistic')
                if envAssembler is not None:
                    offspring.setProperty('environments',
                                          envAssembler.assemble(molecules, cell),
                                          extension='atomistic')
                structure = offspring.getProperty('structure', extension='atomistic')
                minDistMatrix = self.bondUtility.getDistances(
                    structure.getAtomTypes(), self.conditions.externalPressure)
                if self.simpleMoleculeUtility.checkMinDistances(offspring, minDistMatrix):
                    self.conditions.putConditions(offspring)
                    if self.bondUtility.isConnected(structure):
                        return offspring,

            failCounter += 1


    def molecularCrystal(self, dim, nsym, symbols, numIons, lattice):
        toPymatgen = self.atomistic.AtomicStructureRepresentation.toPymatgenMolecule
        molecules = [self.simpleMoleculeUtility.molecules[symbol] for symbol in np.repeat(symbols, numIons)]
        random_crystal = molecular_crystal(
            dim=dim,
            group=1,
            molecules=[toPymatgen(molecule) for molecule in molecules],
            numMols=None,
            factor = self.factor,
            lattice=None#Lattice.from_matrix(lattice)
        )
        return [molecule.createAtNewCoordinates(mol_site.get_mol_object().cart_coords)
                for mol_site, molecule in zip(random_crystal.mol_sites, molecules)],\
            random_crystal.lattice.matrix, random_crystal.volume


def parseIntSet(nputstr=""):
  selection = set()
  invalid = set()
  # tokens are comma seperated values
  tokens = [x.strip() for x in nputstr.split(',')]
  for i in tokens:
     try:
        # typically tokens are plain old integers
        selection.add(int(i))
     except:
        # if not, then it might be a range
        try:
           token = [int(k.strip()) for k in i.split('-')]
           if len(token) > 1:
              token.sort()
              # we have items seperated by a dash
              # try to build a valid range
              first = token[0]
              last = token[len(token)-1]
              for x in range(first, last+1):
                 selection.add(x)
        except:
           # not an int and not a range...
           invalid.add(i)
  return selection

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def generate2Dcell(nsym, LayerThickness, LayerArea):
    # Depending on the symmetry group chosen, construct a random cell which would fit to the geometrical
    # constraints of layer thickness and area
    if nsym in [1, 3, 11, 12, 13, 23, 24, 25, 26, 49, 55, 56, 65, 69, 70, 73, 77]:
        cParam = LayerThickness
    else:
        cParam = LayerThickness*0.5

    aParam = 1.0
    alphaAngle = 90.0

    if nsym in range(1, 8):
        bParam = np.random.uniform(0.25*aParam, 4.0*aParam)
        gammaAngle = np.random.uniform(20.0, 160.0)

    elif nsym in range(8, 49):
        bParam = np.random.uniform(0.25*aParam, 4.0*aParam)
        gammaAngle = 90.0

    elif nsym in range(49, 65):
        bParam = aParam
        gammaAngle = 90.0

    elif nsym in range(65, 81):
        bParam = aParam
        gammaAngle = 120.0

    if (nsym in [1, 2]) or (nsym in range(8, 19)):
        alphaAngle = np.random.uniform(60.0, 120.0)
        cParam /= np.sin(np.pi / 180.0 * alphaAngle)

    S = np.abs(aParam * bParam * np.sin(np.pi / 180.0 * gammaAngle))
    aParam = aParam / np.sqrt(S) * np.sqrt(LayerArea)
    bParam = bParam / np.sqrt(S) * np.sqrt(LayerArea)

    lat = Lattice.from_para(aParam, bParam, cParam, alphaAngle, 90.0, gammaAngle, PBC=[1, 1, 0])

    return lat

def generate1Dcell(nsym, CylinderRadius, CylinderLength):
    # Depending on the symmetry group chosen, construct a random cell which would fit to the geometrical
    # constraints of radius and cylinder length
    if nsym == 1:
        gammaAngle = np.random.uniform(20.0, 160.0)
        aParam = np.random.uniform(0.4 * CylinderRadius, 1.6 * CylinderRadius)
        if gammaAngle <= 90.0:
            bParam = -aParam * np.cos(np.pi / 180.0 * gammaAngle) \
                     + np.sqrt(
                -aParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + 4.0 * CylinderRadius ** 2.0)
        else:
            bParam = aParam * np.cos(np.pi / 180.0 * gammaAngle) \
                     + np.sqrt(
                -aParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + 4.0 * CylinderRadius ** 2.0)
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, gammaAngle, PBC=[0, 0, 1])

    elif nsym in [2]:
        gammaAngle = np.random.uniform(20.0, 160.0)
        if np.random.uniform(0.0, 1.0) < 0.5:
            aParam = np.random.uniform(0.2 * CylinderRadius, 0.8 * CylinderRadius)
            bParam = min([-aParam * np.cos(np.pi / 180.0 * gammaAngle) \
                          + np.sqrt(
                -aParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + CylinderRadius ** 2.0), \
                          CylinderRadius])
        else:
            bParam = np.random.uniform(0.2 * CylinderRadius, 0.8 * CylinderRadius)
            aParam = min([-bParam * np.cos(np.pi / 180.0 * gammaAngle) \
                          + np.sqrt(
                -bParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + CylinderRadius ** 2.0), \
                          CylinderRadius])
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, gammaAngle, PBC=[0, 0, 1])

    elif nsym in [4, 5]:
        aParam = np.random.uniform(0.2 * CylinderRadius, 0.8 * CylinderRadius)
        bParam = np.sqrt(4.0 * CylinderRadius ** 2.0 - 4.0 * aParam ** 2.0)
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, 90.0, PBC=[0, 0, 1])

    elif nsym in [10]:
        gammaAngle = np.random.uniform(20.0, 160.0)
        aParam = np.random.uniform(0.4 * CylinderRadius, 1.6 * CylinderRadius)
        if gammaAngle <= 90.0:
            bParam = -aParam * np.cos(np.pi / 180.0 * gammaAngle) \
                     + np.sqrt(
                -aParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + 4.0 * CylinderRadius ** 2.0)
        else:
            bParam = aParam * np.cos(np.pi / 180.0 * gammaAngle) \
                     + np.sqrt(
                -aParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + 4.0 * CylinderRadius ** 2.0)
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, gammaAngle, PBC=[0, 0, 1])

    elif nsym in [8, 9, 11, 12]:
        gammaAngle = np.random.uniform(20.0, 160.0)
        if np.random.uniform(0.0, 1.0) < 0.5:
            aParam = np.random.uniform(0.2 * CylinderRadius, 0.8 * CylinderRadius)
            bParam = min([-aParam * np.cos(np.pi / 180.0 * gammaAngle) \
                          + np.sqrt(
                -aParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + CylinderRadius ** 2.0), \
                          CylinderRadius])
        else:
            bParam = np.random.uniform(0.2 * CylinderRadius, 0.8 * CylinderRadius)
            aParam = min([-bParam * np.cos(np.pi / 180.0 * gammaAngle) \
                          + np.sqrt(
                -bParam ** 2.0 * np.sin(np.pi / 180.0 * gammaAngle) ** 2.0 + CylinderRadius ** 2.0), \
                          CylinderRadius])
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, gammaAngle, PBC=[0, 0, 1])

    elif nsym in [3, 18, 19]:
        aParam = np.random.uniform(0.4 * CylinderRadius, 1.6 * CylinderRadius)
        bParam = 0.5 * np.sqrt(4.0 * CylinderRadius ** 2.0 - aParam ** 2.0)
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, 90.0, PBC=[0, 0, 1])

    elif nsym in [6, 7, 13, 14, 15, 16, 17, 20, 21, 22]:
        aParam = np.random.uniform(0.2 * CylinderRadius, 0.8 * CylinderRadius)
        bParam = np.sqrt(CylinderRadius ** 2.0 - aParam ** 2.0)
        lat = Lattice.from_para(aParam, bParam, CylinderLength, 90.0, 90.0, 90.0, PBC=[0, 0, 1])

    elif nsym in range(23, 42):
        lat = Lattice.from_para(CylinderRadius / np.sqrt(2), CylinderRadius / np.sqrt(2), CylinderLength, 90.0,
                                90.0, 90.0, PBC=[0, 0, 1])

    elif nsym in range(42, 76):
        lat = Lattice.from_para(CylinderRadius, CylinderRadius, CylinderLength, 90.0, 90.0, 120.0,
                                PBC=[0, 0, 1])

    return lat

def convertStruc(structurePyxtal, pbc, symbols, LOCAL_VACUUM):

    dim = np.sum(pbc)

    if dim == 3:
        ase_struc = structurePyxtal.to_ase()

    if dim == 2:
        ase_struc = structurePyxtal.to_ase()
        ase_struc.rotate(ase_struc.cell[0], (1, 0, 0), rotate_cell=True)
        ase_struc.rotate([0.0, ase_struc.cell[1, 1], ase_struc.cell[1, 2]], (0, 1, 0), rotate_cell=True)
        zsize = np.amax(ase_struc.get_positions()[:, 2]) - np.amin(ase_struc.get_positions()[:, 2])
        ase_struc.cell[2,:] = [0.0, 0.0, zsize + LOCAL_VACUUM]
        ase_struc.center(axis = 2, about = (zsize + LOCAL_VACUUM) / 2.0)
        ase_struc.wrap(pbc=[1, 1, 0])

    if dim == 1:
        ase_struc = structurePyxtal.to_ase()
        xsize = np.amax(ase_struc.get_positions()[:,0])-np.amin(ase_struc.get_positions()[:,0])
        ysize = np.amax(ase_struc.get_positions()[:,1])-np.amin(ase_struc.get_positions()[:,1])
        zsize = ase_struc.cell[2,2]
        tmp_cell = np.array([[xsize + LOCAL_VACUUM, 0.0, 0.0], [0.0, ysize + LOCAL_VACUUM, 0.0],
                            [0.0, 0.0, zsize]])
        ase_struc.set_cell(tmp_cell)
        ase_struc.center(about=(tmp_cell[0, 0] / 2.0, tmp_cell[1, 1] / 2.0, tmp_cell[2, 2] / 2.0))
        ase_struc.wrap(pbc=[0, 0, 1])

    if dim == 0:
        ase_struc = structurePyxtal.to_ase()
        xsize = np.amax(ase_struc.get_positions()[:, 0])-np.amin(ase_struc.get_positions()[:, 0])
        ysize = np.amax(ase_struc.get_positions()[:, 1])-np.amin(ase_struc.get_positions()[:, 1])
        zsize = np.amax(ase_struc.get_positions()[:, 2])-np.amin(ase_struc.get_positions()[:, 2])
        tmp_cell = np.array([[xsize + LOCAL_VACUUM, 0.0, 0.0], [0.0, ysize + LOCAL_VACUUM, 0.0], [0.0, 0.0, zsize + LOCAL_VACUUM]])
        ase_struc.set_cell(tmp_cell)
        ase_struc.center(about=(tmp_cell[0, 0] / 2.0, tmp_cell[1, 1] / 2.0, tmp_cell[2, 2] / 2.0))

    tmp_cell = np.array(ase_struc.cell[:])
    candidate = ase_struc.get_scaled_positions()
    if (pbc == (1, 0, 1)) or (pbc == (0, 1, 0)):
        tmp_cell[[1,2],:] = tmp_cell[[2,1],:]
        candidate[:,[1,2]] = candidate[:,[2,1]]
    if (pbc == (0, 1, 1)) or (pbc == (1, 0, 0)):
        tmp_cell[[0,1,2],:] = tmp_cell[[2,0,1],:]
        candidate[:,[0,1,2]] = candidate[:,[2,0,1]]
    ase_nat = ase_struc.get_global_number_of_atoms()
    ase_symb = ase_struc.get_chemical_symbols()
    operations = []
    operation = np.eye(4, dtype=float)
    for s in symbols:
        tmp_operations = []
        for i in range(ase_nat):
            if ase_symb[i] == s:
                operation[0:3, 3] = np.array([candidate[i]])
                tmp_operations.append(np.copy(operation))
        operations.append([[tmp_operations]])

    return tmp_cell, operations
