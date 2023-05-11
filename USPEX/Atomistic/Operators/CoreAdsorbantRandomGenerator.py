import logging
logger = logging.getLogger(__name__)

from time import time
import numpy as np
from scipy.special import binom

MAX_CORELIGAND_TIME = 30
MAX_RANDOM_TIME = 300
MAX_CORELIGAND_ATTEMPTS = 1000
MAX_ROTATION_ATTEMPTS = 20  # rotation step will be 2pi/MAX_ROTATION_ATTEMPTS
MAX_SITE_SAMPLES_TRY = 1000


class CoreAdsorbantRandomGenerator:
    def __init__(self, utilities, debug = False):
        self.cellUtility = utilities.cellUtility
        self.adsorbantUtility = utilities.adsorbantUtility
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.all_angles = 2 * np.pi * np.arange(MAX_ROTATION_ATTEMPTS) / MAX_ROTATION_ATTEMPTS
        self.cell = self.cellUtility.cellType.initFromCellParameters((0, 0, 0))

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
                tmp_molecules = []
                tmp_offspring = {}
                adsorptionmap = set()
                for adsName, quantity in composition.items():
                    adsorbant = self.adsorbantUtility.adsorbants[adsName]
                    compatibleSites = select_compatible_sites(adsName, adsTypesSitesDiGraph)

                    # check if there is enough sites
                    if len(compatibleSites) < quantity:
                        raise Exception(f"Too few sites for {adsName}")

                    max_samples_try = int(min(MAX_SITE_SAMPLES_TRY, binom(len(compatibleSites), quantity)))
                    i_sample = 0
                    isDocked = False
                    while i_sample <= max_samples_try and not isDocked:
                        sample_molecules = tmp_molecules.copy()
                        sites_sample_attempt = np.random.choice(compatibleSites, quantity, replace=False)
                        i_sample += 1
                        for site in sites_sample_attempt:
                            for angle in list(np.random.permutation(self.all_angles)):
                                dock_attempt = site.dock(adsorbant.site, angle)
                                tmp_offspring, isDocked = self.checkDocking(sample_molecules, dock_attempt, npCoreAssembler)
                                if isDocked:
                                    sample_molecules.append(dock_attempt)
                                    break  # from angles loop, to the next site
                                # bad angle, going to next one
                        if isDocked:  # all sites in the sample are docked,
                            tmp_molecules = sample_molecules
                            for site in sites_sample_attempt:
                                adsTypesSitesDiGraph.remove_node(site)
                            break  # we can go the next adsName
                        # else -- bad sample, going to the next one
                    if not isDocked:
                        raise Exception(f"Can't dock {adsName} after {max_samples_try} tries")
                return tmp_offspring,
            except Exception as e:
                logger.debug(e, exc_info=True)
            failCounter += 1

    def checkDocking(self, tmp_molecules, ads_attempt, npCoreAssembler):
        docked = False
        tmp_offspring = {'molecules': tmp_molecules + [ads_attempt], 'cell': self.cell,
                         'environment': npCoreAssembler.assemble(tmp_molecules + [ads_attempt])}
        tmp_struct, _ = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(
            **tmp_offspring)
        tmp_minDistMatrix = self.bondUtility.getDistances(tmp_struct.getAtomTypes(),
                                                          self.conditions.externalPressure)
        tmp_atomDistances = tmp_struct.getAllDistances()
        np.fill_diagonal(tmp_atomDistances, 10.)
        if np.all(tmp_atomDistances >= tmp_minDistMatrix):  # check if docking is good
            self.conditions.putConditions(tmp_offspring)
            structure, disassembler = self.simpleMoleculeUtility.atomicDisassemblerType.assemble(
                **tmp_offspring)
            if self.bondUtility.isConnected(structure):
                docked = True
        return tmp_offspring, docked

def select_compatible_sites(ligand, adsTypesSitesDiGraph):
    return list([adsTypesSitesDiGraph.successors(y) for y in adsTypesSitesDiGraph.successors(ligand)][0])