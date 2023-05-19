import numpy as np
import logging
import networkx as nx
import alphashape


logger = logging.getLogger(__name__)


from ..JunctionUtility import Site, JunctionType


class NanoparticleCore:
    """
    Class NanoparticleCore provides basic functionality for Core-Adsorbant search.
    It consists of the following classes:
    Assembler -- the factory that creates the NanoparticleCore objects and has the utilities that create, store and
    select docking Sites, store the history of processed combinations of core sites and adsorbants
    """
    structureRepresentation = None
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls,representationType, structureType, atomType, cellType, atomicDisassemblerType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.structureRepresentation = representationType
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, structure, sites=None, isFixed: bool = True, **kwargs):
        self._structure = structure
        self.seenAdsorptions = []  # [{site1: ads1, site2:ads12, ...}, ...]
        self.badAdsorptions = []
        self._sitesByType = {}
        if sites is None:
            self.sites = []
        else:
            for site in sites:
                site['junctionTypes'] = frozenset(
                    [JunctionType(jt) for jt in site['junctionTypes']])
            self.sites = [Site(id=idx, **site) for idx, site in enumerate(sites)]
            for site in self.sites:
                for junctionType in site.junctionTypes:
                    if junctionType in self._sitesByType:
                        self._sitesByType[junctionType].append(site)
                    else:
                        self._sitesByType[junctionType] = [site]
        self.isFixed = isFixed
        if self.isFixed:
            self._indices = np.arange(len(structure))
        else:
            self._indices = np.array([], dtype=int)
        self._alphaShapesCollection = {}  # {adsRadius: alphashape}
        self._adsJuncSiteGraph = None

    def getCell(self):
        return self._structure.getCell()

    def getStructure(self):
        return self._structure

    def __repr__(self):
        return f"<CoreAssembler {self._structure.getFormula()}>"

    def assemble(self, molecules, **kwargs):
        wholeSysStruct, _ = NanoparticleCore.atomicDisassemblerType.assemble(molecules + [self._structure],
                                                                       cell=self._structure.getCell())
        newCell = self.getCell().getEnvelopeCell(wholeSysStruct.getCartesianCoordinates())
        newEnvStructure = NanoparticleCore.structureType(self._structure.getAtomTypes(),
                                                           self._structure.getCartesianCoordinates(),
                                                           newCell)
        return [(newEnvStructure, self._indices)]

    def getSitesByType(self, junctionType):
        if junctionType in self._sitesByType:
            return self._sitesByType[junctionType]
        elif junctionType.label in ('FACE', 'EDGE', 'VERTEX'):
            return self.calcAlphashapeSites(junctionType)
        else:
            logger.warning(f'No sites of type "{junctionType}" on the nanoparticle core')

    def calcAlphashapeSites(self, junctionType, mountPointOffset="covalent"):
        # determine active centers + normal vectors self.activeCenters = [(xyz, normal), ...],

        newSites = []

        if junctionType.junctionParam not in self._alphaShapesCollection:
            alpha = 1 / (junctionType.junctionParam +
                         np.max([at.covalent_radius for at in self._structure.getAtomTypes()]))
            self._alphaShapesCollection[junctionType.junctionParam] = \
                alphashape.alphashape(self._structure.getCartesianCoordinates(), alpha=alpha)
        alphaShape = self._alphaShapesCollection[junctionType.junctionParam]
        nSites = len(self.sites)
        if junctionType.label == "FACE":
            newSites = [
                Site(id=idx, mountPoint=m, orientation=v, junctionTypes={junctionType})
                for m, v, idx in zip(alphaShape.triangles_center, alphaShape.face_normals,
                                     range(nSites, nSites + len(alphaShape.triangles_center)))]
        elif junctionType.label == "VERTEX":
            newSites = [
                Site(id=idx, mountPoint=m, orientation=v, junctionTypes={junctionType})
                for m, v, idx in zip(alphaShape.vertices, alphaShape.vertex_normals,
                                     range(nSites, nSites + len(alphaShape.vertices)))]
        elif junctionType.label == "EDGE":
            edgeSites = []
            for adj_e, adj_f, idx in zip(alphaShape.face_adjacency_edges, alphaShape.face_adjacency,
                                         range(nSites, nSites + len(alphaShape.face_adjacency))):
                origin = 0.5 * (alphaShape.vertices[adj_e[0]] + alphaShape.vertices[adj_e[1]])
                normal = alphaShape.face_normals[adj_f[0]] + alphaShape.face_normals[adj_f[1]]
                normal /= np.linalg.norm(normal)
                edgeSites.append(Site(id=idx, mountPoint=origin, orientation=normal, junctionTypes={junctionType}))
            newSites = edgeSites
        [site.doOffset(self.getStructure()) for site in newSites]
        self._sitesByType[junctionType] = newSites
        self.sites += newSites
        return newSites

    def passivateSite(self, site):
        pass

    def getAdsJuncSiteGraph(self, molSitesMapping):
        """
        Directed tripartite graph (adsorbants)-(junctiontypes)-(sites)
        @param adsorbants:
        @return:
        """
        if self._adsJuncSiteGraph is None:
            DG = nx.DiGraph()
            allAdsJuncType = set()
            for adsName, sites in molSitesMapping.items():
                for site in sites:
                    for jt in site.junctionTypes:
                        DG.add_edge(adsName, jt)
                        allAdsJuncType |= {jt}
            for jt in allAdsJuncType:
                sites = self.getSitesByType(jt)
                for site in sites:
                    DG.add_edge(jt, site)
            self._adsJuncSiteGraph = DG
        return self._adsJuncSiteGraph.copy()

    def addSeenAdsorbtion(self, adsMap):
        self.seenAdsorptions.append(adsMap)

    def addBadAdsorption(self, adsMap):
        self.badAdsorptions.append(adsMap)

    def isMapAlreadySeen(self, adsMap):
        return adsMap in self.seenAdsorptions

    def isBadAdsMap(self, adsMap):
        for adsBadMap in self.badAdsorptions:
            if adsBadMap.issubset(adsMap):
                return True
        return False

    @staticmethod
    def build(filename, **kwargs):
        structure = NanoparticleCore.structureRepresentation.readXYZ(filename)
        environment = dict(
            structure=structure,
        )
        return environment
