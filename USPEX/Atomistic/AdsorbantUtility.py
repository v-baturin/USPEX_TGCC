
class Adsorbant:

    def __init__(self, structure, mountpoint, orientation, junctionType=None):
        self.structure = structure
        self.mountpoint = mountpoint
        self.orientation = orientation
        self.junctionType = junctionType
        self.radius = self._calcRadius()

    def _calcRadius(self):
        pass



class AdsorbantUtility(object):
    def __init__(self, adsorbants):
        self.adsorbants = {key: Adsorbant(**adsorbant) for key, adsorbant in adsorbants.items()}
