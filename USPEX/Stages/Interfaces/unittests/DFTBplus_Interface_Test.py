import unittest
import numpy as np
from pathlib import Path

from ....components import AtomisticRepresentation, DFTBplus_Interface

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'dftbSpecific'
GATHEREDPATH = HOMEPATH/'dftbGatheredData'

class DFTBplus_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in dftb_in.hsd does not make sense.
        # Only output will be parsed and properties checked
        interface = DFTBplus_Interface(tag='0', dftb_input=SPECIFICPATH/'dftb_in.hsd_1', kresol=0.04)
        structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        system = dict(
            ID=ID,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
            ase={'pbc': (1, 1, 1)},
            externalPressure=0.0
        )
        results = interface.readOutput(system=system, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(results['enthalpy'], -395.547))