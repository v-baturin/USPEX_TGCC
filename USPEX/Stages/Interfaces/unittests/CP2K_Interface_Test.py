import os
import shutil
import unittest
import filecmp
from os.path import join as pj
import numpy as np

from ....components import AtomisticRepresentation, CP2K_Interface


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
SPECIFICPATH = pj(HOMEPATH, 'cp2kSpecific')
GATHEREDPATH = pj(HOMEPATH, 'cp2kGatheredData')

class CP2K_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in cp2k.inp does not make sense.
        # Only output will be parsed and properties checked
        interface = CP2K_Interface(tag='0', cp2k_in=pj(SPECIFICPATH, 'cp2k.inp_1'), kresol=0.12)
        structure = AtomisticRepresentation.readPOSCAR(pj(GATHEREDPATH, f'input/system{ID}.vasp'), (1, 1, 1))
        system = dict(
            ID=ID,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
            ase={'pbc': (1, 1, 1)},
            externalPressure=0.0
        )
        results = interface.readOutput(system=system, calcFolder=pj(GATHEREDPATH, f'output/CalcFold{ID}'))
        self.assertTrue(np.isclose(results['enthalpy'], -404.816))