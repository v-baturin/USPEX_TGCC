import unittest
import os
from os.path import join as pj
import json
import numpy as np
from itertools import combinations_with_replacement

from ...components import AtomisticRepresentation, EnvironmentUtility, AdsorbantUtility, AtomicDisassembler, Cell


class EnvironmentUtility_TestNanoparticleCore(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        cls.TEST_FILES_DIR = os.path.join(cls.CURRENT_DIR, 'core_adsorbant_files')
        cls.data_NDI = json2dict(pj(cls.TEST_FILES_DIR, 'data_NDI.json'))
        cls.data_Alpha = json2dict(pj(cls.TEST_FILES_DIR, 'data_Alpha.json'))
        cls.NDI_core = cls.compile_core_envutility(cls, cls.data_NDI)
        cls.Alpha_core = cls.compile_core_envutility(cls, cls.data_Alpha)
        cls.adsorbantsNDI = cls.compile_adsorbants(cls, cls.data_NDI)
        cls.adsorbantsAlpha = cls.compile_adsorbants(cls, cls.data_Alpha)

    def compile_adsorbants(self, data_core_adsorbant):
        adsorb_dict = {x.pop('name'): x for x in data_core_adsorbant['adsorbants']}
        for k, v in adsorb_dict.items():
            v['structure'] = AtomisticRepresentation.readXYZ(pj(self.TEST_FILES_DIR, v['filename']))
        return AdsorbantUtility(adsorb_dict)

    def compile_core_envutility(self, data_core_adsorbant):
        for env_descr in data_core_adsorbant["environments"]:
            env_descr['structure'] = AtomisticRepresentation.readXYZ(pj(self.TEST_FILES_DIR, env_descr['filename']))
        return EnvironmentUtility(data_core_adsorbant["environments"])

    def test_dock_NDI(self):
        assembler = self.NDI_core.assemblers[0]
        adsorbant = self.adsorbantsNDI.adsorbants['X2']
        new_ads_struct = assembler.sites[0].dock(adsorbant.site)
        structure, disassembler = AtomicDisassembler.assemble(molecules=[new_ads_struct],
                                                              cell=Cell.initFromCellVectors((0,0,0)),
                                                              environment=assembler.assemble())
        AtomisticRepresentation.writeXYZ(pj(self.TEST_FILES_DIR, 'outNDI.xyz'), structure)

    def test_dock_Alpha(self):
        assembler = self.Alpha_core.assemblers[0]
        adsorbant = self.adsorbantsAlpha.adsorbants['phenyl']
        jtype, = adsorbant.site.junctionTypes
        new_ads_struct = assembler.sites[0].dock(adsorbant.site)
        structure, disassembler = AtomicDisassembler.assemble(molecules=[new_ads_struct],
                                                              cell=Cell.initFromCellVectors((0, 0, 0)),
                                                              environment=assembler.assemble())
        AtomisticRepresentation.writeXYZ(pj(self.TEST_FILES_DIR, 'out.xyz'), structure)






def json2dict(fname):
    with open(fname, 'r') as f:
        jsdata = json.load(f)
    return jsdata

