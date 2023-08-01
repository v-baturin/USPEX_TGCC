import logging
logger = logging.getLogger(__name__)

from time import time
import numpy as np
from scipy.special import binom

MAX_CORELIGAND_TIME = 30
MAX_RANDOM_TIME = 300
MAX_CORELIGAND_ATTEMPTS = 1000
TOTAL_ROTATION_STEPS = 20  # rotation step will be 2pi/MAX_ROTATION_ATTEMPTS
MAX_SITE_SAMPLES_TRY = 1000


class CoreAdsorbantRandomGenerator:
    def __init__(self, utilities, debug = False):
        self.junctionUtility = utilities.junctionUtility
        self.environmentUtility = utilities.environmentUtility
        self.cellUtility = utilities.cellUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.compositionSpace = utilities.compositionSpace
        self.simpleMoleculeUtility = utilities.simpleMoleculeUtility
        self.cellType = type(self.cellUtility.getRandomCell(1, np.empty(0)))
        self.angle_indices = np.arange(TOTAL_ROTATION_STEPS)

        if debug:
            logger.setLevel(logging.DEBUG)

    def __call__(self, offspringFactory=None):
        # Choice of active centers
        # Reorienting adsorbants according to chosen active centers in core
        composition = self.compositionSpace.randomComposition()  # {symbol : numbers, ...}

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
                npCoreAssembler = np.random.choice(self.environmentUtility.environments)
                adsTypesSitesDiGraph = npCoreAssembler.getAdsJuncSiteGraph(self.junctionUtility.molSitesMapping)
                tmp_molecules = []
                tmp_offspring = {}
                goodAdsorptionmap = set()
                for adsName, quantity in composition.items():
                    adsorbant = self.simpleMoleculeUtility.molecules[adsName]
                    adsSites = self.junctionUtility.molSitesMapping[adsName]
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
                            for k_angle in list(np.random.permutation(self.angle_indices)):
                                angle = 2 * np.pi * k_angle / TOTAL_ROTATION_STEPS
                                sampleAdsorptionMap = {(site.id, adsName, k_angle) for site in sites_sample_attempt}
                                if npCoreAssembler.isBadAdsMap(sampleAdsorptionMap | goodAdsorptionmap):
                                    continue

                                dockingTransfmn = site.dockTransformation(np.random.choice(adsSites), angle)
                                dock_attempt = dockingTransfmn.transform(adsorbant)
                                tmp_offspring, isDocked = self.checkDocking(sample_molecules, dock_attempt,
                                                                            npCoreAssembler, offspringFactory)
                                if isDocked:
                                    sample_molecules.append(dock_attempt)
                                    break  # from angles loop, to the next site
                                npCoreAssembler.addBadAdsorption(goodAdsorptionmap | sampleAdsorptionMap)
                                # bad angle, going to next one

                        # done with this sample
                        # add mapping to previously seen

                        if isDocked:  # all sites in the sample are docked,
                            goodAdsorptionmap |= sampleAdsorptionMap
                            tmp_molecules = sample_molecules
                            for site in sites_sample_attempt:
                                adsTypesSitesDiGraph.remove_node(site)
                            break  # we can go the next adsName

                        logger.debug(f"Attempt {i_sample}: bad sample, moving to next one")
                    if not isDocked:
                        raise Exception(f"Can't dock {adsName} after {max_samples_try} tries")
                adsMapString = ' '.join([
                    f"(site {x[0]}, {x[1]}, {int(360 * x[2] / TOTAL_ROTATION_STEPS)}\N{DEGREE SIGN})"
                    for x in goodAdsorptionmap])
                if not npCoreAssembler.isMapAlreadySeen(goodAdsorptionmap):
                    npCoreAssembler.addSeenAdsorbtion(goodAdsorptionmap)
                    logger.debug(f"Adsorption {adsMapString} sucessfully created")
                    tmp_offspring.setProperty('adsorption_map', goodAdsorptionmap)
                    return tmp_offspring,
                else:
                    logger.debug(f"Adsorption {adsMapString} already seen")
                logger.debug(f"Core-Adsorbant generator failed")
            except Exception as e:
                logger.debug(e, exc_info=True)
            failCounter += 1

    def checkDocking(self, tmp_molecules, ads_attempt, npCoreAssembler, offspringFactory):
        docked = False
        cell = self.cellType.initFromCellParameters((0, 0, 0))
        system = {
            'atomistic.molecules': tmp_molecules + [ads_attempt],
            'atomistic.cell': cell,
            'atomistic.environments': npCoreAssembler.assemble(tmp_molecules + [ads_attempt])}
        tmp_offspring = offspringFactory(**system)
        tmp_struct = tmp_offspring.getProperty('structure', prefix='atomistic', suffix='origin')
        tmp_minDistMatrix = self.bondUtility.getDistances(tmp_struct.getAtomTypes(),
                                                          self.conditions.externalPressure)
        # tmp_atomDistances = tmp_struct.getAllDistances()
        # np.fill_diagonal(tmp_atomDistances, 10.)
        # if np.all(tmp_atomDistances >= tmp_minDistMatrix):  # check if docking is good
        if self.simpleMoleculeUtility.checkMinDistances(tmp_offspring, tmp_minDistMatrix):
            self.conditions.putConditions(tmp_offspring)
            if self.bondUtility.isConnected(tmp_struct):
                docked = True
        return tmp_offspring, docked




def select_compatible_sites(ligand, adsTypesSitesDiGraph):
    return list([adsTypesSitesDiGraph.successors(y) for y in adsTypesSitesDiGraph.successors(ligand)][0])