import logging
logger = logging.getLogger(__name__)

import signal
from time import time
import numpy as np

from pyxtal.crystal import Lattice
from pyxtal import pyxtal

MAX_PYXTAL_TIME = 30
MAX_RANDOM_TIME = 300
MAX_PYXTAL_ATTEMPTS = 20
LOCAL_VACUUM = 0.2

class RandSymPyXtal:
    def __init__(self, utilities, nsym=None):
        self.cellUtility = utilities.cellUtility
        self.world = utilities.world
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.ionDistances = utilities.ionDistances
        self.conditions = utilities.conditions
        if self.simpleMoleculeUtility.isTrueMolecular:
            raise RuntimeError("RandSymPyXtal does not currently work in molecular regime.")
        if isinstance(nsym, int):
            self.nsym = [nsym]
        elif isinstance(nsym, str):
            self.nsym = list(parseIntSet(nsym))
        else:
            self.nsym = nsym

    def __call__(self, *args, **kwargs):
        composition = self.compositionSpace.randomComposition()

        symbols = list(composition.keys())
        numIons = list(composition.values())
        estimatedVolume = self.cellUtility.getCellVolume()
        if estimatedVolume is None:
            elementalComposition = self.simpleMoleculeUtility.getElementalComposition(composition)
            estimatedVolume = self.ionDistances.volumeEstimator.calcCompositionVolume(elementalComposition,
                                                                                      self.conditions.externalPressure)

        if np.sum(numIons) == 0:
            raise RuntimeError("Structure with no atoms requested. Skip.")

        # PyXtal cannot work with numIons == 0, so remove the corresponding elements
        for i in reversed(range(len(symbols))):
            if numIons[i] == 0:
                symbols.pop(i)
                numIons.pop(i)

        startTime = time()
        signal.signal(signal.SIGALRM, signal_handler)
        failCounter = 0
        while True:
            endTime = time()
            failedTime = endTime - startTime
            if failCounter > MAX_PYXTAL_ATTEMPTS or failedTime > MAX_RANDOM_TIME:
                raise RuntimeError("RandSymPyXtal failed.")

            #randcell is an auxiliary cell object to get required info from
            elementalComposition = self.simpleMoleculeUtility.getElementalComposition(composition)
            randcell = self.cellUtility.getRandomCell(estimatedVolume, sum(numIons))

            if self.cellUtility.getDim() == 3:

                if self.nsym is None:
                    self.nsym = list(range(1, 231))
                nsym, = np.random.choice(self.nsym, 1)
                logger.debug(f"Trying {nsym} symmetry")

                structurePyxtal = pyxtal()
                signal.alarm(MAX_PYXTAL_TIME)
                try:
                    structurePyxtal.from_random(3, nsym, symbols, numIons)
                except Exception as e:
                    logger.debug(e, exc_info=True)
                signal.alarm(0)

            elif self.cellUtility.getDim() == 2:

                if self.nsym is None:
                    self.nsym = list(range(1, 81))
                nsym, = np.random.choice(self.nsym, 1)
                logger.debug(f"Trying {nsym} symmetry")

                LayerThickness = self.cellUtility.getThickness()
                LayerArea = randcell.getArea()

                structurePyxtal = pyxtal()
                lat = generate2Dcell(nsym, LayerThickness, LayerArea)
                signal.alarm(MAX_PYXTAL_TIME)
                try:
                    structurePyxtal.from_random(2, nsym, symbols, numIons, lattice=lat)
                except Exception as e:
                    logger.debug(e, exc_info=True)
                signal.alarm(0)

            elif self.cellUtility.getDim() == 1:

                if self.nsym is None:
                    self.nsym = list(range(1, 76))
                nsym, = np.random.choice(self.nsym, 1)
                logger.debug(f"Trying {nsym} symmetry")

                CylinderRadius = self.cellUtility.getThickness() / 2.0
                CylinderLength = randcell.getLength()

                structurePyxtal = pyxtal()
                lat = generate1Dcell(nsym, CylinderRadius, CylinderLength)
                signal.alarm(MAX_PYXTAL_TIME)
                try:
                    structurePyxtal.from_random(1, nsym, symbols, numIons, lattice=lat)
                except Exception as e:
                    logger.debug(e, exc_info=True)
                signal.alarm(0)

            elif self.cellUtility.getDim() == 0:

                if self.nsym is None:
                    self.nsym = list(range(1, 57))
                nsym, = np.random.choice(self.nsym, 1)
                logger.debug(f"Trying {nsym} symmetry")

                structurePyxtal = pyxtal()
                signal.alarm(MAX_PYXTAL_TIME)
                try:
                    structurePyxtal.from_random(0, nsym, symbols, numIons)
                except Exception as e:
                    logger.debug(e, exc_info=True)
                signal.alarm(0)

            if structurePyxtal.valid:
                tmp_cell, coordinates, operations = convertStruc(structurePyxtal, randcell.getPBC(),
                                                                 symbols, LOCAL_VACUUM)
                cell = self.cellUtility.adjustCell(tmp_cell, estimatedVolume, sum(numIons))
                coordinates = dict(zip(symbols, coordinates))
                operations = dict(zip(symbols, operations))
                molecules = self.simpleMoleculeUtility.populateStructure(cell, coordinates, operations)
                atomSymbols, atomDistances = self.simpleMoleculeUtility.getMinDistances(molecules, cell)
                minDistMatrix = self.ionDistances.getDistances(atomSymbols, self.conditions.externalPressure)
                if np.all(atomDistances >= minDistMatrix):
                    system = {'molecules': molecules, 'cell': cell}
                    self.world.putEnvironment(system)
                    self.conditions.putConditions(system)
                    return (system,)

            failCounter += 1

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
    coordinates = []
    for s in symbols:
        tmp_coordinates = []
        for i in range(ase_nat):
            if ase_symb[i] == s:
                tmp_coordinates.append(np.array([candidate[i]]))
        coordinates.append(tmp_coordinates)

    return tmp_cell, coordinates, [None] * len(coordinates)
