import numpy as np
from .Transformation import Transformation
import logging

ALPHA_JUNCTION_LABELS = ['VERTEX', 'EDGE', 'FACE']


class JunctionType:
    def __init__(self, label, junctParam=None):
        self.label = label
        self.junctionParam = junctParam

    def __repr__(self):
        if isinstance(self.junctionParam, float):
            paramstr = f", {self.junctionParam:.2f}"
        elif self.junctionParam is not None:
            paramstr = f", {self.junctionParam}"
        else:
            paramstr = ""
        return f"<junctionType {self.label}{paramstr}>"

    def __eq__(self, other):
        try:
            isParamEqual = (np.abs(self.junctionParam - other.junctionParam) < 1e-6)
        except TypeError:
            isParamEqual = (self.junctionParam == other.junctionParam)
        return (self.label == other.label) and isParamEqual

    def __hash__(self):
        return hash((self.label, self.junctionParam))

class Site:
    """
    Class Site describes docking site
    The object stores information of the host structure, mountPoint and the orientation.
    orientation vector points OUTWARDS the structure

    The object has method dock(self, other, ownAxisAngle=0), that returns the structure of the "other" site, so that
    the "mountPoint"s coincide and orientations FACE each other (--> <--). Also rotation of the "other" by angle
    ownAxisAngle around its orientation is performed

    """

    def __init__(self, mountPoint, orientation, junctionTypes=None, passivateBy=None, id=0):
        self.id = id
        self.orientation = np.array(orientation)
        self.mountPoint = np.array(mountPoint)
        self.junctionTypes = junctionTypes if junctionTypes else None  # frozenset([NanoparticleCore.JunctionType(jt) for jt in junctionTypes]) \
        self.passivateBy = passivateBy

    def __repr__(self):
        return f"<Site #{self.id} {self.host}, junctionTypes={self.junctionTypes}>"

    def dockTransformation(self, other, otherAxisAngle=0):
        assert self.junctionTypes & other.junctionTypes
        shiftOriginToMountpoint = Transformation.fromRotVector((0., 0., 0.), -other.mountPoint)
        rotAroundOrientationAxis = Transformation.fromRotVector(-other.orientation * otherAxisAngle, 0.)
        # rotated_structure =\
        #     rotAroundOrientationAxis.transform(shiftOriginToMountpoint.transform(other.host.getStructure()))
        rot_ax = np.cross(-other.orientation, self.orientation)
        rot_ax /= np.linalg.norm(rot_ax)
        alpha = np.arccos(-other.orientation @ self.orientation)
        matchOrientationTransform = Transformation.fromRotVector(alpha * rot_ax, self.mountPoint)
        netTransform = matchOrientationTransform * (rotAroundOrientationAxis * shiftOriginToMountpoint)
        return netTransform

    def __hash__(self):
        return hash(tuple(map(tuple, (self.mountPoint, self.orientation, self.junctionTypes))))

    def doOffset(self, structure, mountPointOffset="covalent"):
        """
        Shifts mountPoint so that it's located not too close to the host structure. the distance is either
        user-defined (mountPointOffset=R), or equals to covalent atomic radius
        Let A -- atomic position
            e -- unit vector (self.self.orientation with proper dimensions)
            M -- self.mountPoint
            R -- radius of sphere around A, that we don't penetrate
        Objective:
            out of two points:
                Q1, Q2 -- intersections of a line (M,e) with a sphere (A, R)
            find the one that has the largest coordinate along e and update M to that point
        Code performs this operation to all atoms in vectorized fashion and finds the maximum shift along e
        @param structure:
        @param mountPointOffset: float or "covalent"
        @return:
        """
        e = self.orientation.reshape((1, -1))
        AM = self.mountPoint - structure.getCartesianCoordinates()
        AMx = (AM @ e.T)  # projection of AM onto e
        eAMx = AMx @ e  # component of AM along e
        if mountPointOffset == "covalent":
            radii = np.array([x.covalent_radius for x in structure.getAtomTypes()])
        else:
            radii = mountPointOffset
        shift_coeff = np.nanmax(-AMx.T[0] + np.sqrt(radii ** 2 - np.linalg.norm(AM - eAMx, axis=1) ** 2))
        if not np.isnan(shift_coeff) and shift_coeff > 0.:
            self.mountPoint = self.mountPoint + self.orientation * shift_coeff


class JunctionUtility:

    def __init__(self, molSitesMapping=None):
        self.hasJunctions = bool(molSitesMapping)
        self.molSitesMapping = dict()
        for molSymbol, sitesDescriptions in molSitesMapping.items():
            self.molSitesMapping[molSymbol] = [Site(**siteDescription) for siteDescription in sitesDescriptions]
        self._adsorbantsJunctionTypes = None
        self._adsorbantsByJunctionsType = {}

    @staticmethod
    def calculateJunctionTypes(structure, junctionsDescription):

        if set(junctionsDescription) & set(ALPHA_JUNCTION_LABELS):
            molRadius = JunctionUtility._calcEffectiveRadius(structure)
        properClassJunctionTypes = []
        for jt in junctionsDescription:
            if jt in ALPHA_JUNCTION_LABELS:
                properClassJunctionTypes.append(JunctionType(label=jt, junctParam=molRadius))
            else:
                properClassJunctionTypes.append(JunctionType(label=jt, junctParam=None))
        return frozenset(properClassJunctionTypes)


    @staticmethod
    def _calcEffectiveRadius(structure):
        distMatrix = structure.getAllDistances()
        geometricalDiameter = np.max(distMatrix)
        diametralAtomsIdx = np.where(distMatrix == geometricalDiameter)[0]
        maxAtRadius = np.max([at.covalent_radius for at in structure.getAtomTypes()[diametralAtomsIdx]])
        return geometricalDiameter / 2 + maxAtRadius


__author__ = "Vladimir Baturin"
