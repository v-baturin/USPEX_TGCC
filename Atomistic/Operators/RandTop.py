import logging
logger = logging.getLogger(__name__)


import itertools
import json
import numpy as np
import os
import pandas
import time

from sympy.combinatorics.partitions import Partition, RGS_rank

from ...SpaceGroups.TopologicalNet import TopologicalNet
from ...SpaceGroups.SpaceGroups3D import Group


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
with open(f'{HOMEPATH}/idealnets.json', 'rt') as f:
    TOPOLOGICAL_NETS = pandas.DataFrame.from_dict(json.load(f)).T

MAX_SUPERSIZE = 4


class RandTop:
    def __init__(self, cellUtility, compositionSpace, radialDistributionUtility,
                 supercells: list = None, maxSupersize: int = MAX_SUPERSIZE):
        self.cellUtility = cellUtility
        self.compositionSpace = compositionSpace
        self.radialDistributionUtility = radialDistributionUtility
        self.supercells = supercells
        self.maxSupersize = maxSupersize
        self.arxiv = {}

    def __call__(self, *args, **kwargs):
        composition = self.compositionSpace.randomComposition()

        symbols = list(composition.keys())
        numIons = list(composition.values())
        inds = np.flatnonzero(numIons)
        zeroInds = np.sort(list(set(range(len(numIons))) - set(inds)))
        # generateStructureWithRandomTopology properly works if all elements in numIons are nonzero.
        # inds are indices of nonzero elements of numIons.
        numberOfAtoms = [numIons[i] for i in inds]
        totalAtomNubmber = np.sum(numberOfAtoms)
        appropriateNets = TOPOLOGICAL_NETS[totalAtomNubmber % TOPOLOGICAL_NETS['totalAtomNumber'] == 0]
        compstart = time.perf_counter()
        for name, params in appropriateNets.sample(min(appropriateNets.shape[0], 100)).iterrows():
            compend = time.perf_counter()
            if compend - compstart > 60:
                break
            supersize = totalAtomNubmber // params['totalAtomNumber']
            if supersize > self.maxSupersize:
                continue
            net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])
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
                    if len(flavour.nodes) >= len(numberOfAtoms):
                        atomPermutations = list(itertools.permutations(enumerate(numberOfAtoms)))
                        for nodePartition in randomPartitionSampler(len(flavour.multiplicities), len(numberOfAtoms), 50):
                            logger.debug(f'Trying {nodePartition} partition')
                            for atoms3 in randomPermutation(atomPermutations):
                                permutationAtoms, numberOfAtoms = zip(*atoms3)
                                numberOfNodes = [np.sum(flavour.multiplicities[np.asarray(nodeTypeGroup, dtype=int)])
                                                 for nodeTypeGroup in nodePartition]
                                if np.all(np.asarray(numberOfNodes) == np.asarray(numberOfAtoms)):
                                    coordinates = []
                                    operations = []
                                    for atomNumber in np.argsort(permutationAtoms):
                                        nodeIndices = np.asarray(list(nodePartition)[atomNumber], dtype=np.int)
                                        coordinates.append(flavour.group(flavour.nodes[nodeIndices]))
                                        operations.append([flavour.operations[ind] for ind in nodeIndices])
                                    cell = np.asarray(params['cell']) * np.asarray(supercell)

                                    # zeroInds are indices of zero elements of numIons.
                                    # Here we reconstruct numIons consistent arrays by inserting empty arrays
                                    # at zeroInds sites of arrays created with generateStructureWithRandomTopology.
                                    for ind in zeroInds:
                                        coordinates.insert(ind, [])
                                        operations.insert(ind, [])
                                    all_coordinates = np.vstack([*itertools.chain(*coordinates)])
                                    if name in self.arxiv:
                                        for arxivCoordinates in self.arxiv[name]:
                                            if (all_coordinates.shape == arxivCoordinates.shape) and \
                                                    np.allclose(all_coordinates, arxivCoordinates):
                                                continue
                                    if name in self.arxiv:
                                        self.arxiv[name].append(all_coordinates)
                                    else:
                                        self.arxiv[name] = [all_coordinates]

                                    cell = self.cellUtility.adjustCell(cell)
                                    operations = dict(zip(symbols, operations))

                                    molecules = self.compositionSpace.populateStructure(cell, operations)

                                    return ({'molecules' : molecules, 'cell': cell},)
        raise RuntimeError("Operator failed.")


def randomPermutation(array, enumerate = False, maxSize = None):
    maxSize = maxSize if maxSize is not None else len(array)
    for i in np.random.permutation(list(range(len(array))))[0:maxSize]:
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
