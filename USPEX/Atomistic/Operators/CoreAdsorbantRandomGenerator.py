import logging
logger = logging.getLogger(__name__)
import numpy as np


class CoreAdsorbantRandomGenerator:
    def __init__(self, utilities, debug = False):
        self.cellUtility = utilities.cellUtility
        self.adsorbantUtility = utilities.adsorbantUtility
        self.environmentUtility = utilities.environmentUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        self.compositionSpace = utilities.compositionSpace
        if debug:
            logger.setLevel(logging.DEBUG)

    def __call__(self, *args, **kwargs):
        # Choice of active centers
        # Reorienting adsorbants according to chosen active centers in core
        composition = self.compositionSpace.randomComposition()  # {symbol : numbers, ...}
        npCoreAssembler = np.random.choice(self.environmentUtility.assemblers)
        adsTypesSitesDiGraph = npCoreAssembler.getAdsJuncSiteGraph(self.adsorbantUtility.adsorbants)
        current_adsorption = dict()
        molecules = []
        for ads, quantity in composition.items():
            compatibleSites = list([adsTypesSitesDiGraph.successors(y) for y in adsTypesSitesDiGraph.successors(ads)][0])
            selectedSites = np.random.choice(compatibleSites, quantity)
            for site in selectedSites:
                molecules.append(site.dock(self.adsorbantUtility.adsorbants[ads]))
                adsTypesSitesDiGraph.remove_node(site)
