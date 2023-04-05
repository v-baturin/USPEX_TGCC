import os
from pathlib import Path
import numpy as np
from ase.io import read

xligands_dir = Path('ligand_X')
yligands_dir = Path('ligand_Y')
core_file = Path('core_SONY.xyz')

def find_anchor_atom(atoms):
