# http://gitlab.skoltech.ru/shapeev/mlip-dev/blob/master/src/external/python/mlippy/cfgs.py


import numpy as np

from ase import Atoms


def readcfg(filename):
    with open(filename, 'r') as f:
        lat = np.zeros((3, 3))
        types = None
        pos = None
        energy = None
        forces = None
        stresses = None
        size = -1
        mode = -1
        line = f.readline()
        while line:
            line = line.upper()
            line = line.strip()
            if mode == 0:
                if line.startswith('SIZE'):
                    line = f.readline()
                    size = int(line.strip())
                    types = np.zeros(size)
                    pos = np.zeros((size, 3))
                elif line.startswith('SUPERCELL'):
                    line = f.readline()
                    vals = line.strip().split()
                    lat[0, :] = vals[0:3]
                    line = f.readline()
                    vals = line.strip().split()
                    lat[1, :] = vals[0:3]
                    line = f.readline()
                    vals = line.strip().split()
                    lat[2, :] = vals[0:3]
                elif line.startswith('ATOMDATA'):
                    if line.endswith('FZ'):
                        forces = np.zeros((size, 3))
                    for i in range(size):
                        line = f.readline()
                        vals = line.strip().split()
                        types[i] = vals[1]
                        pos[i, :] = vals[2:5]
                        if forces is not None:
                            forces[i, :] = vals[5:8]
                elif line.startswith('ENERGY'):
                    line = f.readline()
                    energy = float(line.strip())
                elif line.startswith('PLUSSTRESS'):
                    line = f.readline()
                    vals = line.strip().split()
                    stresses = np.zeros(6)
                    stresses[:] = vals[0:6]
            if line.startswith('BEGIN_CFG'):
                mode = 0
            elif line.startswith('END_CFG'):
                break
            line = f.readline()

    atoms = Atoms(positions=pos, cell=lat, numbers=types, pbc=[True, True, True])
    if energy is not None:
        atoms.energy = energy
    if forces is not None:
        atoms.forces = forces
    if stresses is not None:
        atoms.stresses = stresses

    return atoms


def savecfg(filename, atoms):
    with open(filename, 'w') as f:
        atstr1 = 'AtomData:  id type      cartes_x      cartes_y      cartes_z           fx          fy          fz\n'
        atstr2 = 'AtomData:  id type      cartes_x      cartes_y      cartes_z\n'
        size = len(atoms.numbers)
        f.write('BEGIN_CFG\n')
        f.write('Size\n')
        f.write(f'   {size}\n')
        f.write('SuperCell\n')
        for i in range(3):
            f.write(' %13f %13f %13f\n' % (atoms.cell[i, 0], atoms.cell[i, 1], atoms.cell[i, 2]))
        if hasattr(atoms, 'forces'):
            f.write(atstr1)
        else:
            f.write(atstr2)
        for i in range(size):
            if hasattr(atoms, 'forces'):
                f.write('         %4d %4d %13f %13f %13f %11.8e %11.8e %11.8e\n' %
                        (i + 1, atoms.numbers[i], atoms.positions[i, 0], atoms.positions[i, 1], atoms.positions[i, 2],
                         atoms.forces[i, 0], atoms.forces[i, 1], atoms.forces[i, 2]))
            else:
                f.write('         %4d %4d %13f %13f %13f\n' %
                        (i + 1, atoms.numbers[i], atoms.positions[i, 0], atoms.positions[i, 1], atoms.positions[i, 2]))
        if hasattr(atoms, 'energy'):
            f.write(' Energy\n   %20f\n' % atoms.energy)
        if hasattr(atoms, 'stresses'):
            f.write(' PlusStress:  xx           yy           zz           yz           xz           xy\n')
            f.write('         %11f %11f %11f %11f %11f %11f\n' %
                    (atoms.stresses[0], atoms.stresses[1], atoms.stresses[2],
                     atoms.stresses[3], atoms.stresses[4], atoms.stresses[5]))
        f.write('END_CFG\n')
