import argparse
from ase.io import read as ase_read
from ase.io import write as ase_write


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--numseeds", required=False, default=50, help='No of first structures to be put to seeds')
ap.add_argument("-f", "--POSCARfile", required=False, default='results*', help='Source poscar file')
input_kwargs = vars(ap.parse_args())
