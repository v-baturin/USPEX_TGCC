import unittest
import numpy as np
from pathlib import Path

from ..CP2K_Interface import CP2K_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'cp2kSpecific'
GATHEREDPATH = HOMEPATH/'cp2kGatheredData'

class CP2K_InterfaceTest(unittest.TestCase):

    def test_read_output(self):
        ID = 0
        # HERE what is written in cp2k.inp does not make sense.
        # Only output will be parsed and properties checked
        interface = CP2K_Interface(tag='0', cp2k_in=SPECIFICPATH/'cp2k.inp_1', kresol=0.12,
                                      targetProperties=['structure', 'enthalpy'])
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
        intermediate = disassembler.disassemble(structure)
        intermediate['.externalPressure'] = 0.0
        intermediate['atomistic.disassembler'] = disassembler
        intermediate = EntryFlavour(extensions=extensions, **intermediate)
        result = interface.readOutput(intermediate, calcFolder=GATHEREDPATH/f'output/CalcFold{ID}')
        self.assertTrue(np.isclose(result['.enthalpy'], -404.816))