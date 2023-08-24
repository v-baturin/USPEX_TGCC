import unittest
import numpy as np
from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ....components import AtomicStructureRepresentation, XTB_Interface, Atomistic

HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'xtbSpecific'
GATHEREDPATH = HOMEPATH/'xtbGatheredData'

class XTB_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in xtb.inp does not make sense.
        # Only output will be parsed and properties checked
        interface = XTB_Interface(tag='0', xtb_input=SPECIFICPATH/'xtb.inp_1')
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (0, 0, 0))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        system = PoolEntry(ID, EntryFlavour(extensions=extensions))
        system.setProperty('externalPressure', 0.0)
        system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
        system.setProperty('disassembler', disassembler, extension='atomistic', suffix='0')
        system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')

        interface.readOutput(system=system, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(system['.enthalpy.0'], -1003.116))