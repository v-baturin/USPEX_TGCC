import unittest
from pathlib import Path
import json
import numpy as np

from ....components import AtomicStructureRepresentation, NanoparticleCore, JunctionUtility, Cell


class NanoparticleCore_Test(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = Path(__file__).parent
        cls.TEST_FILES_DIR = cls.CURRENT_DIR/'core_adsorbant_files'
        cls.data_NDI = json2dict(cls.TEST_FILES_DIR/'data_NDI.json')
        cls.data_Alpha = json2dict(cls.TEST_FILES_DIR/'data_Alpha.json')
        cls.NDI_core = cls.compile_core(cls.data_NDI)
        cls.Alpha_core = cls.compile_core(cls.data_Alpha)
        cls.moleculesNDI, cls.adsorbantsNDI = cls.compile_adsorbants(cls.data_NDI)
        cls.moleculesAlpha, cls.adsorbantsAlpha = cls.compile_adsorbants(cls.data_Alpha)

    @classmethod
    def compile_adsorbants(self, data_core_adsorbant):
        molSitesMapping = dict()
        molecules = dict()
        for ads in data_core_adsorbant['adsorbants']:
            adsName = ads['name']
            structure = AtomicStructureRepresentation.readXYZ(self.TEST_FILES_DIR/ads['filename'])
            molecules[adsName] = structure
            adsSites = []
            for site in ads['sites']:
                site['junctionTypes'] = \
                    JunctionUtility.calculateJunctionTypes(structure,
                                                           junctionsDescription=site['junctionTypes'])
                adsSites.append(site)
            molSitesMapping[adsName] = adsSites

        return molecules, JunctionUtility(molSitesMapping=molSitesMapping)

    @classmethod
    def compile_core(cls, data_core_adsorbant):
        core_descr = data_core_adsorbant["core"]
        core_descr['structure'] = AtomicStructureRepresentation.readXYZ(cls.TEST_FILES_DIR/core_descr['filename'])
        return NanoparticleCore(**core_descr)

    def test_dock_NDI(self):
        adsName = 'Y2'
        assembler = self.NDI_core
        adsorbantSites = self.adsorbantsNDI.molSitesMapping[adsName]
        adsorbantmol = self.moleculesNDI[adsName]
        dockingTransf = assembler.sites[2].dockTransformation(adsorbantSites[0], np.pi / 2)
        new_ads_struct = dockingTransf.transform(adsorbantmol)
        system = {'atomistic.molecules': [new_ads_struct],
                  'atomistic.cell': Cell.initFromCellVectors((0, 0, 0)),
                  'atomistic.environments': assembler.assemble([new_ads_struct])
                  }
        structure, disassembler = NanoparticleCore.Atomistic.atomicDisassemblerType.assemble(system)
        AtomicStructureRepresentation.writeXYZ(self.TEST_FILES_DIR/'outNDI.xyz', structure)

    def test_dock_Alpha(self):
        adsName = 'phenyl'
        coreSiteNo = 2
        assembler = self.Alpha_core
        adsorbantSites = self.adsorbantsAlpha.molSitesMapping[adsName]
        adsorbantmol = self.moleculesAlpha[adsName]
        jtype, = adsorbantSites[0].junctionTypes
        assembler.getSitesByType(jtype)
        dockingTransf = assembler.sites[coreSiteNo].dockTransformation(adsorbantSites[0], np.pi / 2)
        new_ads_struct = dockingTransf.transform(adsorbantmol)
        system = {'atomistic.molecules': [new_ads_struct],
                  'atomistic.cell': Cell.initFromCellVectors((0, 0, 0)),
                  'atomistic.environments': assembler.assemble([new_ads_struct])
                  }
        structure, disassembler = NanoparticleCore.Atomistic.atomicDisassemblerType.assemble(system)
        AtomicStructureRepresentation.writeXYZ(self.TEST_FILES_DIR/'outAlpha.xyz', structure)

def json2dict(fname):
    with open(fname, 'r') as f:
        jsdata = json.load(f)
    return jsdata

