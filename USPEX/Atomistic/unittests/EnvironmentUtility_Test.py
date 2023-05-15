import unittest
import os
from os.path import join as pj
import json
import numpy as np
from itertools import combinations_with_replacement

from ...components import AtomisticRepresentation, EnvironmentUtility, JunctionUtility, AtomicDisassembler, Cell


class EnvironmentUtility_TestNanoparticleCore(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.TEST_FILES_DIR = os.path.join(cls.CURRENT_DIR, 'core_adsorbant_files')
        cls.data_NDI = json2dict(pj(cls.TEST_FILES_DIR, 'data_NDI.json'))
        cls.data_Alpha = json2dict(pj(cls.TEST_FILES_DIR, 'data_Alpha.json'))
        cls.NDI_core = cls.compile_core_envutility(cls, cls.data_NDI)
        cls.Alpha_core = cls.compile_core_envutility(cls, cls.data_Alpha)
        cls.moleculesNDI, cls.adsorbantsNDI = cls.compile_adsorbants(cls, cls.data_NDI)
        cls.moleculesAlpha, cls.adsorbantsAlpha = cls.compile_adsorbants(cls, cls.data_Alpha)

    def compile_adsorbants(self, data_core_adsorbant):
        molSitesMapping = dict()
        molecules = dict()
        for ads in data_core_adsorbant['adsorbants']:
            adsName = ads['name']
            structure = AtomisticRepresentation.readXYZ(pj(self.TEST_FILES_DIR, ads['filename']))
            molecules[adsName] = structure
            adsSites = []
            for site in ads['sites']:
                site['junctionTypes'] = \
                    JunctionUtility.calculateJunctionTypes(structure,
                                                           junctionsDescription=site['junctionTypes'])
                adsSites.append(site)
            molSitesMapping[adsName] = adsSites

        return molecules, JunctionUtility(molSitesMapping=molSitesMapping)

    def compile_core_envutility(self, data_core_adsorbant):
        for env_descr in data_core_adsorbant["environments"]:
            env_descr['structure'] = AtomisticRepresentation.readXYZ(pj(self.TEST_FILES_DIR, env_descr['filename']))
        return EnvironmentUtility(data_core_adsorbant["environments"])

    def test_dock_NDI(self):
        adsName = 'Y2'
        assembler = self.NDI_core.assemblers[0]
        adsorbantSites = self.adsorbantsNDI.molSitesMapping[adsName]
        adsorbantmol = self.moleculesNDI[adsName]
        dockingTransf = assembler.sites[2].dockTransformation(adsorbantSites[0], np.pi / 2)
        new_ads_struct = dockingTransf.transform(adsorbantmol)
        structure, disassembler = AtomicDisassembler.assemble(molecules=[new_ads_struct],
                                                              cell=Cell.initFromCellVectors((0,0,0)),
                                                              environment=assembler.assemble([new_ads_struct]))
        AtomisticRepresentation.writeXYZ(pj(self.TEST_FILES_DIR, 'outNDI.xyz'), structure)

    def test_dock_Alpha(self):
        adsName = 'phenyl'
        coreSiteNo = 2
        assembler = self.Alpha_core.assemblers[0]
        adsorbantSites = self.adsorbantsAlpha.molSitesMapping[adsName]
        adsorbantmol = self.moleculesAlpha[adsName]
        jtype, = adsorbantSites[0].junctionTypes
        assembler.getSitesByType(jtype)
        dockingTransf = assembler.sites[coreSiteNo].dockTransformation(adsorbantSites[0], np.pi / 2)
        new_ads_struct = dockingTransf.transform(adsorbantmol)
        structure, disassembler = AtomicDisassembler.assemble(molecules=[new_ads_struct],
                                                              cell=Cell.initFromCellVectors((0, 0, 0)),
                                                              environment=assembler.assemble([new_ads_struct]))
        AtomisticRepresentation.writeXYZ(pj(self.TEST_FILES_DIR, 'outAlpha.xyz'), structure)

def json2dict(fname):
    with open(fname, 'r') as f:
        jsdata = json.load(f)
    return jsdata

