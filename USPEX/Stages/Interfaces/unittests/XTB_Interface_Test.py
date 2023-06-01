import unittest
import numpy as np
from pathlib import Path

from ....components import AtomisticRepresentation, XTB_Interface, AtomisticPoolEntry

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'xtbSpecific'
GATHEREDPATH = HOMEPATH/'xtbGatheredData'

class XTB_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in xtb.inp does not make sense.
        # Only output will be parsed and properties checked
        interface = XTB_Interface(tag='0', xtb_input=SPECIFICPATH/'xtb.inp_1')
        structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (0, 0, 0))
        system = AtomisticPoolEntry(
            ID=ID,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(
                np.arange(len(structure)).reshape((-1, 1))),
            ase={'pbc': (0, 0, 0)},
            externalPressure=0.0
        )
        interface.readOutput(system=system, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(system['enthalpy'], -1003.116))