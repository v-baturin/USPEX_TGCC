import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple
from .EnvironmentUtility import NanoparticleCore
import logging

ALPHA_JUNCTION_LABELS = ['VERTEX', 'EDGE', 'FACE']

JunctionType = NanoparticleCore.Site.JunctionType #namedtuple('AlphaJunctionType', ['label', 'adsRadius'])


class Adsorbant:

    def __init__(self, structure, mountpoint, orientation, junctionTypes=None, filename=None):
        self.structure = structure
        self.mountPoint = mountpoint
        self.orientation = orientation
        if junctionTypes is None:
            logging.info(f"No junctionType specified in {filename}. Trying to use alpha-shape-based sites")
            junctionTypes = ALPHA_JUNCTION_LABELS
        if set(junctionTypes) & set(ALPHA_JUNCTION_LABELS):
            self._r = self._calcEffectiveRadius()
        for k in junctionTypes:
            if junctionTypes[k] in ALPHA_JUNCTION_LABELS:
                junctionTypes[k] = JunctionType(label=junctionTypes[k], adsRadius=self._r)
            else:
                junctionTypes[k] = JunctionType(label=junctionTypes[k], adsRadius=None)
        self.junctionTypes = frozenset(junctionTypes)

    def _calcEffectiveRadius(self):
        distMatrix = cdist(self.structure.getCartesianCoordinates(), self.structure.getCartesianCoordinates())
        geometricalDiameter = np.max(distMatrix)
        diametralAtomsIdx = np.where(distMatrix == geometricalDiameter)[0]
        maxAtRadius = np.max([at.covalent_radius for at in self.structure.getAtomTypes()[diametralAtomsIdx]])
        return geometricalDiameter / 2 + maxAtRadius



class AdsorbantUtility(object):
    def __init__(self, adsorbants):
        self.adsorbants = {key: Adsorbant(**adsorbant) for key, adsorbant in adsorbants.items()}
