import logging
logger = logging.getLogger(__name__)

from time import time
import numpy as np

MAX_CORELIGAND_TIME = 30
MAX_RANDOM_TIME = 300
MAX_CORELIGAND_ATTEMPTS = 1000

class CoreAdsorbantRandomGenerator:
    def __init__(self, utilities, debug = False):
        self.cellUtility = utilities.cellUtility
        self.adsorbantUtility = utilities.adsorbantUtility
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility

        self.cell = self.cellUtility.cellType.initFromCellParameters((0,0,0))

        if debug:
            logger.setLevel(logging.DEBUG)

    def __call__(self, *args, **kwargs):
        # Choice of active centers
        # Reorienting adsorbants according to chosen active centers in core
        composition = self.compositionSpace.randomComposition()  # {symbol : numbers, ...}
        symbols = list(composition.keys())
        numIons = list(composition.values())

        if np.sum(numIons) == 0:
            raise RuntimeError("Structure with no atoms requested. Skip.")
        startTime = time()
        failCounter = 0
        while True:

            endTime = time()
            failedTime = endTime - startTime
            if failCounter > MAX_CORELIGAND_ATTEMPTS or failedTime > MAX_RANDOM_TIME:
                raise RuntimeError("Core-adsorbant generator failed.")

            try:
                npCoreAssembler = np.random.choice(self.environmentUtility.assemblers)
                adsTypesSitesDiGraph = npCoreAssembler.getAdsJuncSiteGraph(self.adsorbantUtility.adsorbants)
                molecules = []
                adsorptionmap = {}
                for adsName, quantity in composition.items():
                    compatibleSites = list([adsTypesSitesDiGraph.successors(y) for y in adsTypesSitesDiGraph.successors(adsName)][0])
                    selectedSites = np.random.choice(compatibleSites, quantity)
                    for site in selectedSites:
                        adsorptionmap[site.id] = adsName
                        molecules.append(site.dock(self.adsorbantUtility.adsorbants[adsName].site, np.pi / 2))
                        adsTypesSitesDiGraph.remove_node(site)
                if adsorptionmap in npCoreAssembler.seenAdsorptions:
                    failCounter += 1
                    continue
                offspring = {'molecules': molecules, 'cell': self.cell,
                             'environment': npCoreAssembler.assemble(molecules)}
                # atomSymbols, atomDistances, disassembler = self.simpleMoleculeUtility.getMinDistances(**offspring)
                # minDistMatrix = self.bondUtility.getDistances(atomSymbols, self.conditions.externalPressure)
                structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(**offspring)
                minDistMatrix = self.bondUtility.getDistances(structure.getAtomTypes(),
                                                              self.conditions.externalPressure)
                atomDistances = structure.getAllDistances()
                np.fill_diagonal(atomDistances, 10.)
                if np.all(atomDistances >= minDistMatrix):
                    self.conditions.putConditions(offspring)
                    structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(**offspring)
                    if self.bondUtility.isConnected(structure):
                        npCoreAssembler.seenAdsorptions.append(adsorptionmap)
                        return offspring,
            except Exception as e:
                logger.debug(e, exc_info=True)

            failCounter += 1
