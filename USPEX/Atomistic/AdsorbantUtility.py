import numpy as np
from scipy.spatial.distance import cdist
from collections import namedtuple
from .EnvironmentUtility import NanoparticleCore
import logging

ALPHA_JUNCTION_LABELS = ['VERTEX', 'EDGE', 'FACE']

JunctionType = NanoparticleCore.JunctionType
Site = NanoparticleCore.Site


class Adsorbant:

    def __init__(self, structure, mountPoint, orientation, junctionTypes=None, filename=None, **kwargs):
        self._structure = structure
        if junctionTypes is None:
            logging.info(f"No junctionType specified in {filename}. Trying to use alpha-shape-based sites")
            junctionTypes = ALPHA_JUNCTION_LABELS
        if set(junctionTypes) & set(ALPHA_JUNCTION_LABELS):
            self._r = self._calcEffectiveRadius()
        properClassJunctionTypes = []
        for jt in junctionTypes:
            if jt in ALPHA_JUNCTION_LABELS:
                properClassJunctionTypes.append(JunctionType(label=jt, junctParam=self._r))
            else:
                properClassJunctionTypes.append(JunctionType(label=jt, junctParam=None))
        junctionTypes = frozenset(properClassJunctionTypes)
        self.site = Site(host=self, mountPoint=mountPoint, orientation=orientation, junctionTypes=junctionTypes)

    def _calcEffectiveRadius(self):
        distMatrix = cdist(self._structure.getCartesianCoordinates(), self._structure.getCartesianCoordinates())
        geometricalDiameter = np.max(distMatrix)
        diametralAtomsIdx = np.where(distMatrix == geometricalDiameter)[0]
        maxAtRadius = np.max([at.covalent_radius for at in self._structure.getAtomTypes()[diametralAtomsIdx]])
        return geometricalDiameter / 2 + maxAtRadius

    def getStructure(self):
        return self._structure


class AdsorbantUtility(object):
    def __init__(self, adsorbants):
        self.adsorbants = {key: Adsorbant(**adsorbant) for key, adsorbant in adsorbants.items()}
        self._adsorbantsJunctionTypes = None
        self._adsorbantsByJunctionsType = {}

    @property
    def adsorbantsByJunctionType(self):
        if not self._adsorbantsByJunctionsType:
            for adsorbant in self.adsorbants:
                for junctionType in adsorbant.site.junctionTypes:
                    if junctionType in self._adsorbantsByJunctionsType:
                        self._adsorbantsByJunctionsType[junctionType].append(adsorbant)
                    else:
                        self._adsorbantsByJunctionsType[junctionType] = [adsorbant]
        return self._adsorbantsByJunctionsType

    @property
    def adsorbantJunctionTypes(self):
        if self._adsorbantsJunctionTypes is None:
            self._adsorbantsJunctionTypes = set()
            for ads in self.adsorbants:
                self._adsorbantsJunctionTypes |= set(ads.junctionTypes)
        return self._adsorbantsJunctionTypes


__author__ = "Vladimir Baturin"
