import logging
logger = logging.getLogger(__name__)


class CoreAdsorbantRandomGenerator:
    def __init__(self, utilities, debug = False):
        self.cellUtility = utilities.cellUtility
        self.environmentUtility = utilities.environmentUtility
        self.adsorbantUtility = utilities.AdsorbantUtility
        self.bondUtility = utilities.bondUtility
        self.conditions = utilities.conditions
        if debug:
            logger.setLevel(logging.DEBUG)

    def __call__(self, *args, **kwargs):
        # Choice of active centers
        # Reorienting adsorbants according to chosen active centers in core
        pass

# Code for development purposes ****

