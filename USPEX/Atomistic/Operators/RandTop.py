import logging
logger = logging.getLogger(__name__)


import itertools
import json
import numpy as np
import pandas
import time
import signal

from pathlib import Path
from sympy.combinatorics.partitions import Partition, RGS_rank

from ..SymmetricStructure import SymmetricStructure
from ...SpaceGroups.SpaceGroups3D import Group


HOMEPATH = Path(__file__).parent
with open(HOMEPATH/'idealnets.json', 'rt') as f:
    TOPOLOGICAL_NETS = pandas.DataFrame.from_dict(json.load(f)).T

MAX_SUPERSIZE = 4
ATTEMPTS_ROTATION = 50
ATTEMPTS_POINT_GROUP = 10
MAX_RANDOM_TIME = 300


class RandTop:
    def __init__(self, utilities, supercells: list = None, maxSupersize: int = MAX_SUPERSIZE,
                 attemptsRotation: int = ATTEMPTS_ROTATION, attemptsPointGroup: int = ATTEMPTS_POINT_GROUP):
        self.cellUtility = utilities.cellUtility
        self.environmentUtility = utilities.environmentUtility
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.supercells = supercells
        self.maxSupersize = maxSupersize
        self.attemptsRotation = attemptsRotation
        self.attemptsPointGroup = attemptsPointGroup
        self.arxiv = {}
        signal.signal(signal.SIGALRM, signal_handler)

    def __setstate__(self, state):
        self.__dict__.update(state)
        signal.signal(signal.SIGALRM, signal_handler)

    def __call__(self, offspringFactory):
        composition = self.compositionSpace.randomComposition()
        envAssembler = np.random.choice(self.environmentUtility.environments) if self.environmentUtility.environments \
            else None
        envCell = envAssembler.getCell() if envAssembler is not None else None

        symbols = list(composition.keys())
        numIons = list(composition[symbol] for symbol in symbols)
        inds = np.flatnonzero(numIons)
        zeroInds = np.sort(list(set(range(len(numIons))) - set(inds)))
        # generateStructureWithRandomTopology properly works if all elements in numIons are nonzero.
        # inds are indices of nonzero elements of numIons.
        numberOfAtoms = [numIons[i] for i in inds]
        totalAtomNumber = int(np.sum(numberOfAtoms))
        appropriateNets = TOPOLOGICAL_NETS[totalAtomNumber % TOPOLOGICAL_NETS['totalAtomNumber'] == 0]
        compstart = time.perf_counter()
        signal.alarm(MAX_RANDOM_TIME)
        try:
            for name, params in appropriateNets.sample(min(appropriateNets.shape[0], 100)).iterrows():
                compend = time.perf_counter()
                if compend - compstart > 60:
                    break
                supersize = totalAtomNumber // params['totalAtomNumber']
                if supersize > self.maxSupersize:
                    continue
                net = SymmetricStructure(name, Group.getGroupFromSymbol(params['groupName']), params['nods'])
                logger.debug(f'Trying {name} topology with {params["groupName"]} symmetry')
                supercells = decompose3(supersize) if self.supercells is None \
                    else [supercell for supercell in self.supercells if np.prod(supercell) == supersize]
                for supercell in randomPermutation(supercells):
                    logger.debug(f'Trying {supercell} supercell')
                    topstart = time.perf_counter()
                    for flavour in randomPermutation(net.flavours(tuple(supercell))):
                        topend = time.perf_counter()
                        if topend - topstart > 15:
                            break
                        if len(flavour.sites) >= len(numberOfAtoms):
                            atomPermutations = list(itertools.permutations(enumerate(numberOfAtoms)))
                            for nodePartition in randomPartitionSampler(len(flavour.multiplicities), len(numberOfAtoms),
                                                                        50):
                                logger.debug(f'Trying {nodePartition} partition')
                                for atoms3 in randomPermutation(atomPermutations):
                                    permutationAtoms, numberOfAtomsPermutated = zip(*atoms3)
                                    numberOfNodes = [
                                        np.sum(flavour.multiplicities[np.asarray(nodeTypeGroup, dtype=int)])
                                        for nodeTypeGroup in nodePartition]
                                    if np.all(np.asarray(numberOfNodes) == np.asarray(numberOfAtomsPermutated)):
                                        coordinates = []
                                        operations = []
                                        for atomNumber in np.argsort(permutationAtoms):
                                            nodeIndices = np.asarray(list(nodePartition)[atomNumber], dtype=int)
                                            coordinates.append(flavour.group(flavour.sites[nodeIndices]))
                                            operations.append([flavour.operations[ind] for ind in nodeIndices])
                                        cell = np.asarray(params['cell']) * np.asarray(supercell)

                                        # zeroInds are indices of zero elements of numIons.
                                        # Here we reconstruct numIons consistent arrays by inserting empty arrays
                                        # at zeroInds sites of arrays created with generateStructureWithRandomTopology.
                                        for ind in zeroInds:
                                            coordinates.insert(ind, [])
                                            operations.insert(ind, [])

                                        estimatedVolume = self.cellUtility.getCellVolume()
                                        if estimatedVolume is None:
                                            elementalComposition = self.simpleMoleculeUtility.getElementalComposition(
                                                composition)
                                            estimatedVolume = self.bondUtility.volumeEstimator.calcCompositionVolume(
                                                elementalComposition,
                                                self.conditions.externalPressure)
                                        cell = self.cellUtility.adjustCell(cell, estimatedVolume, totalAtomNumber,
                                                                           baseCell=envCell)
                                        operations = dict(zip(symbols, operations))
                                        all_coordinates = np.vstack([*itertools.chain(*coordinates)])
                                        attemptsRotation = self.attemptsRotation if self.simpleMoleculeUtility.isTrueMolecular else 1

                                        if name not in self.arxiv:
                                            self.arxiv[name] = []
                                        for arxivCoordinates in self.arxiv[name]:
                                            if (all_coordinates.shape == arxivCoordinates.shape) and \
                                                    np.allclose(all_coordinates, arxivCoordinates):
                                                break
                                        else:
                                            self.arxiv[name].append(all_coordinates)
                                            for i in range(attemptsRotation):
                                                offspring = offspringFactory(
                                                    **self.simpleMoleculeUtility.populateStructure(cell, operations))
                                                molecules = offspring.getProperty('molecules', extension='atomistic')
                                                cell = offspring.getProperty('cell', extension='atomistic')
                                                if len(molecules) != totalAtomNumber:
                                                    continue
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
                                                        signal.alarm(0)
                                                        return offspring,
        except Exception as e:
            logger.debug(e, exc_info=True)
        signal.alarm(0)
        raise RuntimeError("RandTop failed.")


def randomPermutation(array, enumerate = False, maxSize = None):
    maxSize = maxSize if maxSize is not None else len(array)
    for i in np.random.permutation(list(range(len(array))))[0:maxSize]:
        i = int(i)  # this is workaround for numpy int issue
        yield (i, array[i]) if enumerate else array[i]

def randomPartitionSampler(itemsNumber : int, partitionsNumber : int, samplesNumber : int):
    if itemsNumber < partitionsNumber:
        raise ValueError("Number of items should be greater than or equal to number of partitions.")
    usedRGS = []
    attempts = 1000
    while (samplesNumber > len(usedRGS)) and attempts > 0:
        RGS = []
        n = itemsNumber - partitionsNumber
        for i in list(range(partitionsNumber)):
            if i == partitionsNumber - 1:
                k = n
            else:
                k = np.random.randint(n) if n else 0
            n -= k
            RGS.append(i)
            RGS.extend(np.random.randint(i + 1,size=k))
        if RGS_rank(RGS) not in usedRGS:
            usedRGS.append(RGS_rank(RGS))
            yield Partition.from_rgs(RGS, list(range(itemsNumber))).partition
        else:
            attempts -= 1

def decompose3(m123):
    """
    Lists all decompositions of a number into 3 multiplicands.
    :param m123: A number to be decomposed.
    :return: List of 3-tuples.
    """
    decompositions = [(m123, 1, 1)]
    for m1 in list(range(1, m123)):
        if not m123 % m1:
            m23 = m123 // m1
            decompositions.append((m1,m23,1))
            for m2 in list(range(1, m23)):
                if not m23%m2:
                    m3 = m23//m2
                    decompositions.append((m1,m2,m3))
    return decompositions

def signal_handler(signum, frame):
    raise Exception("Timed out!")
