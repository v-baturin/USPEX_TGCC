
class Adsorbant:

    def __init__(self, structure, mountpoint, orientation):
        self.structure = structure
        self.mountpoint = mountpoint
        self.orientation = orientation



class AdsorbantUtility(object):
    def __init__(self, adsorbants):
        self.adsorbants = adsorbants
