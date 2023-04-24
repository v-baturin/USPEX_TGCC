import numpy as np
import subprocess as sp

from pathlib import Path

from ...CellUtility import Cell

from .GetPermutation import GetPermutation
from .GetPrimitiveCell import GetPrimitiveCell
from .Get_Final_Struc import Get_Final_Struc
from .Get_Init_Lattice import Get_Init_Lattice
from .Read_Stokes_output import Read_Stokes_output
from .Write_Stokes_input import Write_Stokes_input
from .fix_latticeStokes_after import fix_latticeStokes_after
from .spaceGroups import SpaceGroups
from .unitCellFromPrimitive import unitCellFromPrimitive

HOMEPATH = Path(__file__).parent


def symope_crystal(CenterminDistMatrice, fixLat, fixRndSeed, nsym, numIons, lat, sym_coef, silent=True):
    """
    The function is for generation of a random cell with a specified space group using Stokes' code.
    :param ORG_STRUC: global input structure.
    :param nsym: space group number.
    :param numIons: number of atoms.
    :param lat: lattice.
    :param sym_coef: symmetry coefficient.
    :return candidate: generated candidate.
    :return Lattice: generated lattice.
    :return errorS: error flag.
    """

    spgBINDIR = HOMEPATH/'spacegroup'  # path to the Stokes' executable

    #fixRndSeed = master._params['fixRndSeed']  # fix rand seed, to reproduce results

    sGroup = SpaceGroups().return_group(nsym)  # space group's standard symbol

    #numIons = numIons_standardize(numIons)

    # Initilizating outputs:
    # 1) initial coordinates
    # 2) initial lattice
    # 3) error check
    candidate = np.zeros((np.sum(numIons), 3))
    Lattice = np.asarray([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=float)
    errorS = 0

    # Flowchart:

    # 1) Permutation axis:
    permutation, permutationBack = GetPermutation(nsym)

    # 2) Get initial lattice:
    Init_lat = Get_Init_Lattice(lat, permutation)

    # 3) Fix non-conventional lattice (disabled in Matlab code):
    # %Init_lat = fix_latticeStokes_before(nsym, Init_lat);

    # 4) Primitive cell:
    Init_numIons, Init_lat, fail = GetPrimitiveCell(sGroup, Init_lat, numIons, fixLat)

    if not fail:
        # 5) Prepare input for Stokes' code:
        stokes_in_content = Write_Stokes_input(Init_numIons, CenterminDistMatrice, nsym, Init_lat, fixRndSeed, sym_coef)

        #if not os.path.exists('data'):
        #    os.mkdir('data')
        #textdata = ['data_space2d.txt', 'data_space.txt', 'data_wyckoff.txt', 'data_diperiodic.txt']
        #for txt in textdata:
        #    source = spgBINDIR + '/data/' + txt
        #    target = 'data/' + txt
        #    if not os.path.exists(target) and os.path.exists(source):
        #        shutil.copy(source, target)

        code_name = 'random_cell'
        proc = sp.Popen(code_name, cwd = spgBINDIR, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True)
        outs, errs = proc.communicate(input=stokes_in_content)
        status = proc.returncode
        #shutil.rmtree('data')

        '''
        # Write all the contents to the input file:
        stokes_in_file = 'rc.in'
        stokes_out_file = 'rc.out'
        with open(stokes_in_file, 'w') as f:
            f.write(stokes_in_content)

        # 6) Execute Stokes' code:
        code_name = 'random_cell'

        if os.name == 'posix':
            execute_locally = True
        else:
            execute_locally = False

        if execute_locally:
            if not os.path.exists('.data'):
                os.mkdir('.data')
            textdata = ['data_space2d.txt', 'data_space.txt', 'data_wyckoff.txt']
            for txt in textdata:
                source = spgBINDIR + '/data/' + txt
                target = '.data/' + txt
                if not os.path.exists(target) and os.path.exists(source):
                    shutil.copy(source, target)
            status = local_execution(code_name, stokes_in_file, stokes_out_file, spgBINDIR, silent=silent)
            if os.path.exists('.data'):
                shutil.rmtree('.data')
        else:
            # Get data from API instead of getting it via local program execution:
            # Here we should invoke web-based API at http://han.ess.sunysb.edu/stokes_symmetry/api.php.
            status = remote_execution(code_name, stokes_in_content, stokes_out_file, silent=silent)
            if status > 0:
                # Process the case of unsuccessful response using local programs:
                status = local_execution(code_name, stokes_in_file, stokes_out_file, spgBINDIR, silent=silent)

        '''

        if not silent:
            print('  Stokes <%s> execution status: %i' % (code_name, status))

        if status == 0:
            # 7) Read Stokes' outputs:
            coordinate_S, lattice_S, failed = Read_Stokes_output(outs)#stokes_out_file

            # 8) Adjust lattice if needed:
            if not failed:
                lattice, coordinate = fix_latticeStokes_after(nsym, lattice_S, coordinate_S)
                #lattice[3:6] *= (np.pi / 180.0)  # go back to radians

                Lattice_Matrix = Cell.initFromCellParameters((1,1,1), *lattice).getCellVectors()
                if fixLat:
                    pass
                    #Lattice_Matrix = latConverter(lattice)
                else:
                    #Lattice_Matrix = latConverter(lattice)
                    abs_cand = np.dot(coordinate, Lattice_Matrix)

                    # 8.1) Optimize lattice:
                    coordinate = np.dot(abs_cand, np.linalg.inv(Lattice_Matrix))

                limit = 0.000001
                Lattice_Matrix[np.where(abs(Lattice_Matrix) < limit)] = 0.0
                coordinate[np.where(abs(coordinate) < limit)] = 0.0
                coordinate[np.where(abs(coordinate - 1.0) < limit)] = 0.0

                # 9) Get conventional cell:

                Lat_type = sGroup[0]

                if Lat_type != 'P' and fixLat:
                    # 9.1) Special groups:
                    if 37 < nsym < 42:
                        Lat_type = 'B'

                    Lattice_Matrix, candidate = unitCellFromPrimitive(Lat_type, Lattice_Matrix, coordinate)
                else:
                    candidate = np.copy(coordinate)

                # 10) Permuation back:
                Lattice, candidate = Get_Final_Struc(Lattice_Matrix, candidate, permutationBack)

                # system = Crystal(symbols=config.chemicalSymbols, cell=Lattice_Matrix)
                #
                # if not config.isGoodLattice(system):
                #     errorS = 1

                # Fix for ticket #152 - Problem with splitBigCell:
                if candidate.shape[0] != np.sum(numIons):
                    errorS = 1

            else:
                errorS = 1
        else:
            errorS = 1
    else:
        errorS = 1

    if errorS == 0:
        return candidate, Lattice
    else:
        raise RuntimeError("Symope failed.")

'''
if __name__ == "__main__":
    from lib.functions import read_input
    from lib.createORGStruc import createORGStruc
    from lib.createCalcFolder import createCalcFolder

    try:
        input_content = read_input('../../INPUT.txt')
    except:
        input_content = read_input('INPUT.txt')

    ORG_STRUC = createORGStruc(input_content)

    # Define the path one level up to allow correct copy of the tools:
    if ORG_STRUC['USPEXPath'].find('lib') >= 0:
        ORG_STRUC['USPEXPath'] = os.path.dirname(os.path.dirname(ORG_STRUC['USPEXPath']))
    createCalcFolder(ORG_STRUC, tmp_type='Atomistic')

    os.chdir(ORG_STRUC['homePath'] + '/CalcFoldTemp')

    # nsym = 227  # F
    # nsym = 146  # R
    # nsym = 39
    nsym = ORG_STRUC['nsym'][0]
    numIons = ORG_STRUC['numIons']
    # lat = [[10, 12, 13, 89.01 / 180. * np.pi, 120.99999 / 180. * np.pi, 90.99 / 180. * np.pi]]  # [[1, 0, 0], [0, 2, 0], [0, 0, 3]] # ORG_STRUC['latVolume'][0][0]
    lat = [[5, 0, 0], [0, 6, 0], [0, 0, 7]]

    # lat = ORG_STRUC['latVolume'][0][0]

    # lat = ORG_STRUC['lattice']

    minD = ORG_STRUC['minDistMatrice']
    sym_coef = ORG_STRUC['sym_coef']
    candidate, newLattice, errorS = symope_crystal(ORG_STRUC, nsym, numIons, lat, minD, sym_coef)

    from pprint import pprint

    for var in [candidate, newLattice, errorS]:
        pprint(var)

    os.chdir(ORG_STRUC['homePath'])
'''