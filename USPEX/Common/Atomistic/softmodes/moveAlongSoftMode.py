import numpy as np

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from USPEX.Common.Atomistic.Element import Element
from .AtomTypeCounter import atomTypeCounter


def move_along_SoftMode(system, GOOD_BONDS, EIGENVECTOR, MUT_DEGREE):
    """
    Atom mutation based on soft modes: all atoms moved along eigenvector corresponding to the softest mode.

    :param system: system.
    :param eigvector: eigenvector.
    :param mut_degree: coeficient that here will be converted to maximal displacement of atoms.
    :return deviation: deviation.
    """

    coord = system.scaled_coordinates
    lattice = system.cell

    N = len(system)
    vec = np.zeros(3)
    new_Coord = np.copy(system.scaled_coordinates)

    close_enough = -0.37 * np.log(GOOD_BONDS)  # this part needed for clusters
    vect = np.zeros(3)
    # Organize an array with a list of numbers of atoms starting from 0 (in Matlab it starts from 1):
    atomTypes, atom_types = atomTypeCounter(system.chemicalSymbols)
    R_val = np.array([Element(atomType).covalent_radius for atomType in atomTypes])

    if system.dimension == 0:  # cluster == 1
        coord += 0.0001
        # TODO: implement later makeCluster() and moveCluster() functions:
        '''
        [lat, candidate] = makeCluster(lattice, coord, ORG_STRUC.vacuumSize(1));
        [candidates] = moveCluster(lat, candidate);
        lattice = lat;
        coord = candidates;
        '''


    coef = 0.0
    for i in range(N):
        vec[0] = EIGENVECTOR[i*3 + 0]
        vec[1] = EIGENVECTOR[i*3 + 1]
        vec[2] = EIGENVECTOR[i*3 + 2]
        coef = max(np.linalg.norm(vec), coef)

    # This way the maximal displacement for an atom is equal to howManyMut*mut_degree:
    normfac = MUT_DEGREE / float(coef)

    notDone = True
    step = 0

    deviation = np.zeros(N)
    while notDone:
        for i in range(N):  # shift the atoms
            vec[0] = EIGENVECTOR[i * 3 + 0]
            vec[1] = EIGENVECTOR[i * 3 + 1]
            vec[2] = EIGENVECTOR[i * 3 + 2]

            new_Coord[i, :] = coord[i, :] + normfac * np.dot(vec, np.linalg.inv(lattice))
            deviation[i] = np.linalg.norm(normfac * vec)

        if system.dimension == 0:  # cluster == 1, if atoms move away form cluster - mutate them less
            badAtoms = np.ones(N)
            dist = 10.0 * np.ones((N, N))
            min_dist = np.zeros(N)
            for i in range(N):
                for j in range(i + 1, N):
                    vect[0] = new_Coord[i, 0] - new_Coord[j, 0]
                    vect[1] = new_Coord[i, 1] - new_Coord[j, 1]
                    vect[2] = new_Coord[i, 2] - new_Coord[j, 2]

                    delta = np.sqrt(np.sum(np.dot(vect, lattice) ** 2) - R_val[at_types[i]] - R_val[at_types[j]])
                    dist[i, j] = dist[j, i] = delta
                    if delta < close_enough[at_types[i], at_types[j]]:
                        badAtoms[i] = 0
                        badAtoms[j] = 0

                min_dist[i] = np.min(dist[i, :])

            if np.sum(badAtoms) == 0:
                notDone = 0
            else:
                step += 1
                for i in range(N):
                    if badAtoms[i]:
                        EIGENVECTOR[i * 3 + 0, 0] *= 0.9
                        EIGENVECTOR[i * 3 + 1, 0] *= 0.9
                        EIGENVECTOR[i * 3 + 2, 0] *= 0.9
        else:
            notDone = False

        if step > 10:
            '''
            %  dist = 10*ones(N,N);
            %  min_dist = zeros(1,N);
            %  for i = 1 : N
            %   for j = i+1 : N
            %     vect(1) = coord(i,1) - coord(j,1);
            %     vect(2) = coord(i,2) - coord(j,2);
            %     vect(3) = coord(i,3) - coord(j,3);
            %     delta = sqrt(sum((vect*lattice).^2)) - R_val(at_types(i)) - R_val(at_types(j));
            %     dist(i,j) = delta;
            %     dist(j,i) = delta;
            %   end
            %   min_dist(i) = min(dist(i,:));
            %  end
            %  dist
            %  min_dist
            %  coord
            %  lattice
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %unix(['echo main_structure >> POSCAR_cluster']);
            %unix(['echo ' '1.0' ' >> POSCAR_cluster']);
            %for latticeLoop = 1:3
            %  unix(['echo ' num2str(lattice(latticeLoop,:)) ' >> POSCAR_cluster']);
            %end
            %unix(['echo ' num2str(numIons) ' >> POSCAR_cluster']);
            %unix(['echo ' 'Direct' ' >> POSCAR_cluster']);
            %for coordLoop = 1:N
            %  unix(['echo ' num2str(coord(coordLoop,:)) ' >> POSCAR_cluster']);
            %end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % quit
            '''
            notDone = False

    newSystem = AtomicStructure(symbols=system.chemicalSymbols, scaled_positions=new_Coord, cell=system.cell,
                                magmoms=system.magmoms)
    return newSystem, deviation


# if __name__ == "__main__":
#     from old.lib.mat2dict import loadmat
#
#     test_dir = 'p301/test_SoftModeMutation_p301_new'
#
#     ORG_STRUC = loadmat(test_dir + '/ORG.mat')['ORG_STRUC']
#
#     coord0 = loadmat(test_dir + '/move_along_SoftMode_Mutation_coord0.mat')['coord0']
#     numIons0 = loadmat(test_dir + '/move_along_SoftMode_Mutation_numIons0.mat')['numIons0']
#     lat0 = loadmat(test_dir + '/move_along_SoftMode_Mutation_lat0.mat')['lat0']
#     eigV = loadmat(test_dir + '/move_along_SoftMode_Mutation_eigV.mat')['eigV']
#
#     lat, new_Coord, deviation = move_along_SoftMode(ORG_STRUC, coord0, numIons0, lat0, eigV, 1)
#
#     print('')
