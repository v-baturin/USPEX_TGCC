__author__ = 'etikhonov, vbaturin'

import numpy as np

def symope_cluster(minDistMatrix, nsym, numIonsFull, rand_cell):
    '''
    Creates structure using symmetry operator nsym\n
    :param nsym: symmetry group
    :param numIonsFull: total number of ions in cluster
    :param lattice: cell
    :param minDistMatrix: IonDistances from INPUT.txt
    :return candidate: generated structure...
    :return newLattice: ...and its lattice
    :return errorS: error status (0 - no errors)
    '''
    minDistMatrix = np.array(minDistMatrix)

    """
    creates the cluster satisfying symmetry nsym
    IMPORTANT: we test the generation of clusters in the ellipse inside the lattice. Making it more 'bulk'
    """
    ellipse_mode = 1
    ###################### Symmetry operations ###################
    E = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])  # equivalence
    I = np.array([[-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])  # inversion
    C2x = np.array([[-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]])  # two fold axes along YOZ
    C2y = np.array([[1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]])  # two fold axes along XOZ
    C2z = np.array([[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, -1.0]])  # two fold axes along XOY
    Hz = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, -1.0]])  # reflection in the mirror plane XOY

    # important! We assign Z as a symmetry axe and then set principal axes as coordinate axes, highest moment of inertia corresponds to Z, lowest - to X
    errorS = 0
    lattice = rand_cell.getCellVectors()
    newLattice = lattice.copy()
    volLat = np.linalg.det(lattice)
    numIons = np.sum(numIonsFull).astype(int)
    nI = numIons
    candidate = []
    if minDistMatrix.size == 1:
        minDistMatrix = np.array(minDistMatrix).reshape((1, 1))
    minDistance = np.max(minDistMatrix)  # CHANGE TO WHOLE ION DISTANCES MATRIX!!!!!!!!!!!!!!!!!!!!!
    if 'E' in nsym:
        candidate = np.random.random((numIons, 3)) - 0.5  # was in Matlab: candidate = rand(sum(numIons),3) - 0.5;
        # but numIons is already result in sum, see above
    elif 'Ci' in nsym or 'S2' in nsym:
        candidate = []
        while 1:
            if numIons == 1:  # odd number of atoms, thus one should be in the dead center
                tmp = np.array([0.0, 0.0, 0.0])
                if len(candidate) > 0:
                    candidate = np.vstack((candidate, tmp))
                else:
                    candidate = tmp.copy()
                break
            if numIons == 0:
                break
            tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
            # and then add (0.5,0.5,0.5)
            while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                tmp = np.random.random((1, 3))
            if abs(tmp.sum()) < 0.01:
                continue
            nextAtom = np.dot(tmp, I)
            cand = np.vstack((tmp, nextAtom))

            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            numIons -= 2
    elif 'I' in nsym or 'i' in nsym:  # icosahedral symmetries
        # easiest way to implement I - icosahedron centered at origin (2-fold axes - X, Z; 3-fold axis - (1,1,1))
        # http://en.wikipedia.org/wiki/File:Icosahedron-golden-rectangles.svg"""
        newLattice = np.diag((volLat ** (1 / 3.),) * 3)
        candidate = []
        if numIons % 2 == 1:  # put in the center
            candidate = np.array([0.0, 0.0, 0.0])
            numIons -= 1
        if numIons < 60 and numIons % 12 != 0 and numIons != 20 and numIons != 30 and numIons != 32:
            if numIons < 40 or numIons == 58 or numIons == 46:
                status = 'Impossible to build the cluster with {0} atoms that has symmetry group {1}'.format(numIons,
                                                                                                             nsym)
                errorS = 1
                with open('error_cluster_symmetry', 'w') as f: f.write(status)
                import sys
                sys.exit(status)
        n60 = 0
        numIons1 = numIons
        while 1:
            numIons1 -= 60
            if numIons1 != 0 and numIons1 < 60:
                if numIons1 != 12 and numIons1 != 20 and numIons1 != 24 and numIons1 != 30 and numIons1 != 32 and numIons1 != 36:
                    if numIons1 < 40 or numIons1 == 58 or numIons1 == 46:
                        break
            else:
                n60 += 1
        GR = (np.sqrt(5) + 1) / 2.0  # Golden Ratio
        alpha = np.arctan(1.0 / GR)  # an angle to turn the axis X around Z to become 5-fold axis!
        R1 = np.array([[np.cos(alpha), - np.sin(alpha), 0.0],
                       [np.sin(alpha), np.cos(alpha), 0.0],
                       [0.0, 0.0, 1.0]])
        R5 = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(2 * np.pi / 5), - np.sin(2 * np.pi / 5)],
                       [0.0, np.sin(2 * np.pi / 5), np.cos(2 * np.pi / 5)]])
        n = 1
        while n <= n60:  # do a 'full' or 'half-full' symmetry point
            tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
            # and then add (0.5,0.5,0.5)
            while np.linalg.norm(tmp) > 0.5 and ellipse_mode:
                tmp = np.random.random((1, 3)) - 0.5
            # tmp = np.array(np.mtrx('{0} -0.33 0'.format(GR))) # for ideal buckyball
            if 'Ih' in nsym or 'ih' in nsym:  # Ih (full icosahedral) symmetry
                if n60 % 2 == 1 and n == n60:  # do only 60 poins :)
                    # (by generating a point on mirror plane)
                    tmp[:, 2] = 0
                else:
                    # do 120 points
                    tmp = np.vstack((tmp, np.dot(tmp, Hz)))  # tmp = cat(1,tmp,tmp*[1 0 0; 0 1 0; 0 0 -1])
                    # in Matlab but latter = Hz
            cand = tmp.copy()
            # rotations around the 5-fold axis (GR, -1, 0)
            """ etikhonov:
            Original MATLAB code there was
             for i = 1 : 4
                nextAtom = tmp*inv(R1)*(R5^i)*R1;
                cand = cat(1,cand,nextAtom);
            end
            """
            for i in range(4):
                if i == 0:
                    R5i = R5.copy()
                else:
                    R5i = np.dot(R5i, R5)
                nextAtom = np.dot(tmp, np.dot(np.linalg.inv(R1), np.dot(R5i, R1)))
                cand = np.vstack((cand, nextAtom))
            # rotations around the main cube diagonal (1,1,1) (3-fold axis)
            nextAtoms1 = np.dot(cand, np.array([[0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0],
                                                [1.0, 0.0, 0.0]]))
            nextAtoms2 = np.dot(cand, np.array([[0.0, 0.0, 1.0],
                                                [1.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0]]))
            cand = np.vstack((cand, nextAtoms1))
            cand = np.vstack((cand, nextAtoms2))
            # rotations around the main axis Z (2-fold axis)
            nextAtoms = np.dot(cand, np.array([[-1.0, 0.0, 0.0],
                                               [0.0, -1.0, 0.0],
                                               [0.0, 0.0, 1.0]]))
            cand = np.vstack((cand, nextAtoms))
            # rotations around the main axis X (2-fold axis)
            nextAtoms = np.dot(cand, np.array(
                [[1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0]]))  # +++ diagonal, -120 degree
            cand = np.vstack((cand, nextAtoms))
            if (('Ih' in nsym) or ('ih' in nsym)) and not ((n60 % 2 == 1) and (n == n60)):
                numIons -= 120
                n += 2
            else:
                numIons -= 60
                n += 1
            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            if n > n60:
                break
        # end
        if numIons > 0:  # numIons = 12x+20y+30z
            for x in range(0, 10 + 1):
                for y in range(0, 6 + 1):
                    for z in range(0, 4 + 1):
                        if 12 * x + 20 * y + 30 * z == numIons:
                            break
                    if 12 * x + 20 * y + 30 * z == numIons:
                        break
                if 12 * x + 20 * y + 30 * z == numIons:
                    break
            for i in range(1, x + 1):  # 12-atoms, thus put them on 5-fold axis
                tmp = np.array([[1.0, -1.0 / GR, 0.0]]) * np.random.random() * 0.5
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.array([[1.0, -1.0 / GR, 0.0]]) * np.random.random() * 0.5
                cand = tmp.copy()
                # rotations around the main cube diagonal (1,1,1) (3-fold axis)
                cand = np.vstack((cand, np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                                              [0.0, 0.0, 1.0],
                                                              [1.0, 0.0, 0.0]]))))
                # +++ diagonal, -120 degree
                cand = np.vstack((cand, np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                                              [1.0, 0.0, 0.0],
                                                              [0.0, 1.0, 0.0]]))))  # -240 degree
                # rotations around the main axis Z (2-fold axis)
                nextAtoms = np.dot(cand, np.array([[-1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0],
                                                   [0.0, 0.0, 1.0]]))  # +++ diagonal, -120 degree
                cand = np.vstack((cand, nextAtoms))
                # rotations around the main axis X (2-fold axis)
                nextAtoms = np.dot(cand, np.array([[1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0]]))  # +++ diagonal, -120 degree
                cand = np.vstack((cand, nextAtoms))
                if len(candidate) == 0:
                    candidate = cand.copy()
                else:
                    candidate = np.vstack((candidate, cand))
            for i in range(1, y + 1):  # 20-atoms, thus put them on 3-fold axis
                tmp1 = np.random.random() * 0.5
                while abs(tmp1) < 0.05:
                    tmp1 = np.random.random() - 0.5
                tmp = np.array([[tmp1, -tmp1, tmp1]])

                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp1 = np.random.random() * 0.5
                    while abs(tmp1) < 0.05:
                        tmp1 = np.random.random() - 0.5
                    tmp = np.array([[tmp1, -tmp1, tmp1]])

                cand = tmp.copy()
                # rotations around the main axis Z (2-fold axis)
                nextAtoms = np.dot(cand, np.array([[-1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0],
                                                   [0.0, 0.0, 1.0]]))  # +++ diagonal, -120 degree
                cand = np.vstack((cand, nextAtoms))
                # rotations around the main axis Y (2-fold axis)
                nextAtoms = np.dot(cand, np.array([[-1.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0],
                                                   [0.0, 0.0, -1.0]]))  # +++ diagonal, -120 degree
                cand = np.vstack((cand, nextAtoms))
                # rotations around the 5-fold axis (GR, -1, 0)
                candTMP = cand.copy()
                for i in range(4):
                    if i == 0:
                        R5i = R5.copy()
                    else:
                        R5i = np.dot(R5i, R5)
                    nextAtom = np.dot(candTMP, np.dot(np.linalg.inv(R1), np.dot(R5i, R1)))
                    cand = np.vstack((cand, nextAtom))
                if len(candidate) == 0:
                    candidate = cand.copy()
                else:
                    candidate = np.vstack((candidate, cand))
            for i in range(1, z + 1):  # 30-atoms, thus put them on 2-fold axis
                tmp1 = np.random.random() - 0.5
                while abs(tmp1) < 0.05:
                    tmp1 = np.random.random() - 0.5
                tmp = np.array([[0, 0, tmp1]])  # Z axis, thus rotate around 2-fold X
                cand = tmp.copy()
                # rotations around the main axis X (2-fold axis)
                nextAtoms = np.dot(cand, np.array([[1.0, 0.0, 0.0],
                                                   [0.0, -1.0, 0.0],
                                                   [0.0, 0.0, -1.0]]))  # +++ diagonal, -120 degree
                cand = np.vstack((cand, nextAtoms))
                # rotations around the main cube diagonal (1,1,1) (3-fold axis)
                nextAtoms1 = np.dot(cand, np.array([[0.0, 1.0, 0.0],
                                                    [0.0, 0.0, 1.0],
                                                    [1.0, 0.0, 0.0]]))  # +++ diagonal, -120 degree
                nextAtoms2 = np.dot(cand, np.array([[0.0, 0.0, 1.0],
                                                    [1.0, 0.0, 0.0],
                                                    [0.0, 1.0, 0.0]]))  # -240 degree
                cand = np.vstack((cand, nextAtoms1))
                cand = np.vstack((cand, nextAtoms2))
                # rotations around the 5-fold axis (GR, -1, 0)
                candTMP = cand.copy()
                for i in range(4):
                    if i == 0:
                        R5i = R5.copy()
                    else:
                        R5i = np.dot(R5i, R5)
                    nextAtom = np.dot(candTMP, np.dot(np.linalg.inv(R1), np.dot(R5i, R1)))
                    cand = np.vstack((cand, nextAtom))
                if len(candidate) == 0:
                    candidate = cand.copy()
                else:
                    candidate = np.vstack((candidate, cand))
    elif 'S' in nsym or 's' in nsym:  # S2n group - 2n rotoreflection (combination rotation+reflection)
        newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                               [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                               [0.0, 0.0, lattice[2][2]]])
        candidate = []
        rotRank = int(nsym[1:])  # should be EVEN (=2n), odd is equivalent to Cnh
        while 1:
            if numIons == 1:
                tmp = np.array([0.0, 0.0, 0.0])
                try:
                    candidate = np.vstack((candidate, tmp))
                except ValueError:
                    candidate = tmp.copy()
                break
            if numIons < rotRank:  # put atoms on rotation axis
                cand = np.array([[0.0, 0.0, np.random.random() - 0.5]])
                for i in range(2, numIons + 1):
                    tmp = np.array([[0.0, 0.0, np.random.random() - 0.5]])  # we work in the space
                    #  [-0,5:0.5;-0,5:0.5;-0,5:0.5] and then add (0.5,0.5,0.5)
                    cand = np.vstack((cand, tmp))
                numIons = 0
            else:
                tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                # and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                    # and then add (0.5,0.5,0.5)
                cand = tmp.copy()
                tooClose = 0

                for i in range(1, rotRank - 1 + 1):
                    if i == 1:
                        Hzi = Hz.copy()
                    else:
                        Hzi = np.dot(Hz, Hzi)
                    angle = 2 * i * np.pi / rotRank
                    opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                       [np.sin(angle), np.cos(angle), 0.0],
                                       [0.0, 0.0, 1.0]])  # rotate
                    # around Z axis
                    nextAtom = np.dot(tmp, np.dot(opemat, (Hzi)))
                    if np.linalg.norm(np.dot((nextAtom - tmp), newLattice)) < minDistance:
                        tooClose = 1
                    cand = np.vstack((cand, nextAtom))
                if tooClose:
                    numIons -= 1
                    cand = np.array([[0.0, 0.0, tmp[0][2]]])
                else:
                    numIons -= rotRank
            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            if numIons == 0:
                break
    elif 'Th' in nsym or 'th' in nsym:
        # Pyritohedral (Th) symmetry
        # easiest way to implement T - include tetrahonal into the cube with O in the center and XYZ parallel to the edges
        newLattice = np.diag((volLat ** (1 / 3.),) * 3)
        candidate = []
        if numIons != 1 and numIons != 6 and numIons != 7 and numIons != 8 and numIons != 9 and numIons < 12:
            status = 'Impossible to build the cluster with {0} atoms that has symmetry group {1}'.format(numIons, nsym)
            with open('error_cluster_symmetry', 'w') as f: f.write(status)
            errorS = 1
        if numIons % 2 == 1:  # put in the center
            candidate = np.array([0.0, 0.0, 0.0])
            numIons -= 1
        n12 = 0
        numIons1 = numIons
        while 1:
            numIons1 -= 12
            if numIons1 != 6 and numIons1 != 8 and numIons1 != 0 and numIons1 < 12:
                break
            else:
                n12 += 1
        n = 1
        while n <= n12:  # do a 'full' or 'half-full' symmetry point
            tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
            # and then add (0.5,0.5,0.5)
            while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                tmp = np.random.random((1, 3)) - 0.5
            if n12 % 2 == 1 and n == n12:
                # do only 12 poins :) (by generating a point on mirror plane)
                tmp[0][np.random.randint(0, 3)] = 0  # MATLAB: tmp(ceil(rand*3)) = 0;
            cand = tmp.copy()
            # 2-fold rotations around X, Y, Z (O is in the cube center, Z goes up)
            nextAtom = np.dot(tmp, np.array([[-1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0],
                                             [0.0, 0.0, 1.0]]))  # rotate around Z axis
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0],
                                             [0.0, 0.0, -1.0]]))  # rotate around X axis
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]]))  # rotate around Y axis
            cand = np.vstack((cand, nextAtom))
            # rotations around the main cube diagonals (3-fold axes)
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0]]))  # +++ diagonal, -120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))  # -240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [-1.0, 0.0, 0.0]]))  # ++- diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [-1.0, 0.0, 0.0]]))  # +-- diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [1.0, 0.0, 0.0]]))  # +-+ diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            if not (n12 % 2 == 1 and n == n12):
                numIons -= 24
                n += 2
                cand1 = -1 * cand  # add an inversion
                cand = np.vstack((cand, cand1))
            else:
                numIons -= 12
                n += 1
            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            if n > n12:
                break
        if numIons > 0:  # numIons can be described as 6x+8y
            # etikhonov:
            # probability distribution was not uniform there! Maybe it was designed, but I made it uniform.
            n1 = np.ceil(numIons / 8.).astype(int)
            n2 = np.floor(numIons / 6.).astype(int)
            # was: n = n1 + np.round(np.random.random() * (n2 - n1)).astype(int)
            # numIons = (4n-numIons/2)*6 + (numIons/2-3n)*8
            n = np.random.randint(n1, n2 + 1)
            x = 4 * n - np.round(numIons / 2.).astype(int)
            y = np.round(numIons / 2.).astype(int) - 3 * n
            for i in range(1, x + 1):  # 6 atoms per go - atoms on the XYZ axes
                tmp = np.random.random() - 0.5
                while abs(tmp) < 0.05:
                    tmp = np.random.random() - 0.5
                candidate = np.vstack((candidate, np.array([[0.0, 0.0, tmp]])))
                candidate = np.vstack((candidate, np.array([[0.0, 0.0, -tmp]])))
                candidate = np.vstack((candidate, np.array([[0.0, tmp, 0.0]])))
                candidate = np.vstack((candidate, np.array([[0.0, -tmp, 0.0]])))
                candidate = np.vstack((candidate, np.array([[tmp, 0.0, 0.0]])))
                candidate = np.vstack((candidate, np.array([[-tmp, 0.0, 0.0]])))
            for i in range(1, y + 1):  # 8 atoms per go - atoms on the main cube diagonals
                tmp = np.random.random() - 0.5
                while abs(tmp) < 0.05 or \
                        (ellipse_mode and (np.linalg.norm(np.array([[tmp, tmp, tmp, ]])) > 0.5)):
                    tmp = np.random.random() - 0.5
                candidate = np.vstack((candidate, np.array([[tmp, tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, -tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, -tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, -tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, -tmp, -tmp]])))
    # TODO: Continue spellcheck from there
    elif 'T' in nsym or 't' in nsym:  # Tetrahedral (T, Td) symmetries
        # easiest way to implement T - include tetrahonal into the cube with O in the center and XYZ parallel to the edges
        newLattice = np.diag((volLat ** (1 / 3.),) * 3)
        candidate = []
        if numIons != 1 and numIons < 4:
            status = 'Impossible to build the cluster with {0} atoms that has symmetry group {1}'.format(numIons, nsym)
            with open('error_cluster_symmetry', 'w') as f: f.write(status)
            errorS = 1
        if numIons % 2 == 1:  # put in the center
            candidate = np.array([0.0, 0.0, 0.0])
            numIons -= 1
        n12 = 0
        numIons1 = numIons
        while 1:
            numIons1 = numIons1 - 12
            if numIons1 != 0 and numIons1 < 4:
                break
            else:
                n12 += 1
        n = 1
        while n <= n12:  # do a 'full' symmetry point
            tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
            # and then add (0.5,0.5,0.5)
            while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                tmp = np.random.random((1, 3)) - 0.5
            if ('Td' in nsym or 'td' in nsym) and (n12 % 2 == 1) and (n == n12):  # Td (full tetrahedral) symmetry
                # do only 12 poins :) (by generating a point on mirror plane)
                tmp[0][0] = tmp[0][1]
            cand = tmp.copy()
            # 2-fold rotations around X, Y, Z (O is in the cube center, Z goes up)
            nextAtom = np.dot(tmp, np.array([[-1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0],
                                             [0.0, 0.0, 1.0]]))  # rotate around X axis
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0],
                                             [0.0, 0.0, -1.0]]))  # rotate around X axis
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0]]))  # rotate around Y axis
            cand = np.vstack((cand, nextAtom))
            # rotations around the main cube diagonals (3-fold axes)
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0]]))  # +++ diagonal, -120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))  # -240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [-1.0, 0.0, 0.0]]))  # ++- diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [-1.0, 0.0, 0.0]]))  # +-- diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [1.0, 0.0, 0.0]]))  # +-+ diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            if ('Td' in nsym or 'td' in nsym) and not (n12 % 2 == 1 and n == n12):  # Td (full tetrahedral) symmetry
                numIons -= 24
                n += 2
                cand1 = cand.copy()  # we add a vertical reflection plane
                #  that goes through Z axis and one of the tetrahedra edges
                cand1[:, 0] = -1 * cand[:, 1]
                cand1[:, 1] = -1 * cand[:, 0]
                cand = np.vstack((cand, cand1))
            else:
                numIons -= 12
                n += 1
            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            if n > n12:
                break
        if numIons > 0:  # numIons can be described as 4x+6y
            n1 = np.ceil(numIons / 6.).astype(int)
            n2 = np.floor(numIons / 4.).astype(int)
            n = n1 + np.round(np.random.random() * (n2 - n1)).astype(
                int)  # numIons = (3n-numIons/2)*4 + (numIons/2-2n)*6
            x = 3 * n - np.round(numIons / 2.).astype(int)
            y = np.round(numIons / 2.).astype(int) - 2 * n
            for i in range(1, x + 1):  # 4 atoms per go - atoms on the main cube diagonals
                tmp = np.random.random() - 0.5
                while abs(tmp) < 0.05 or \
                        (ellipse_mode and (np.linalg.norm(np.array([[tmp, tmp, tmp]])) > 0.5)):
                    tmp = np.random.random() - 0.5
                if len(candidate) == 0:
                    candidate = np.array([[tmp, tmp, tmp]])
                else:
                    candidate = np.vstack((candidate, np.array([[tmp, tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, -tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, -tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, tmp, -tmp]])))
            for i in range(1, y + 1):  # 6 atoms per go - atoms on the XYZ axes
                tmp = np.random.random() - 0.5
                while abs(tmp) < 0.05:
                    tmp = np.random.random() - 0.5
                if len(candidate) == 0:
                    candidate = np.array([[0.0, 0.0, tmp]])
                else:
                    candidate = np.vstack((candidate, np.array([[0.0, 0.0, tmp]])))
                candidate = np.vstack((candidate, np.array([[0.0, 0.0, -tmp]])))
                candidate = np.vstack((candidate, np.array([[0.0, tmp, 0.0]])))
                candidate = np.vstack((candidate, np.array([[0.0, -tmp, 0.0]])))
                candidate = np.vstack((candidate, np.array([[tmp, 0.0, 0.0]])))
                candidate = np.vstack((candidate, np.array([[-tmp, 0.0, 0.0]])))
    elif 'Cv' in nsym or 'cv' in nsym:  # Cnv group - Cn axis and n vertical mirror planes
        rotRank = int(nsym[2:])  # ceil(rand(1)*nsym);
        if rotRank > 2:
            newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                                   [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                                   [0.0, 0.0, lattice[2][2]]])
        candidate = []
        while 1:
            if numIons == 1:
                tmp = np.array([0.0, 0.0, 0.0])
                try:
                    candidate = np.vstack((candidate, tmp))
                except ValueError:
                    candidate = tmp.copy()
                break
            if numIons < 2 * rotRank:
                if numIons < rotRank:  # put atoms on axis
                    cand = np.array([[0.0, 0.0, np.random.random() - 0.5]])
                    for i in range(2, numIons + 1):
                        tmp = np.array([[0.0, 0.0, np.random.random() - 0.5]])  # we work
                        #  in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5] and then add (0.5,0.5,0.5)
                        cand = np.vstack((cand, tmp))
                    numIons = 0
                else:  # put atoms in mirror planes: generate atom randomly then 'rotate' it till it meets the plane
                    tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                    # and then add (0.5,0.5,0.5)
                    while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                        tmp = np.random.random((1, 3)) - 0.5
                    r = np.linalg.norm(tmp)
                    tmp = np.array([[np.sqrt(r ** 2 - tmp[0][2] ** 2), 0.0, tmp[0][2]]])  # mirror plan ZOX
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    numIons -= rotRank
            else:
                tmp = np.random.random(
                    (1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                # and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5
                cand = tmp.copy()
                if (tmp[0][0] * newLattice[0][0]) ** 2 + (tmp[0][1] * newLattice[1][1]) ** 2 < minDistance ** 2:  # too
                    #  close to the main axis
                    numIons -= 1
                    cand = np.array([[0.0, 0.0, tmp[0][2]]])
                else:
                    tooClose = 0
                    for i in range(1, rotRank + 1):  # check if atom is too close to some mirror plane
                        #  (in this case - put atom on a plane and only rotate)
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # reflection
                        #  across a plane that goes through axis Z and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        if np.linalg.norm(np.dot((nextAtom - tmp), newLattice)) < minDistance:
                            tooClose = 1
                            tmp = (tmp + nextAtom) / 2.0
                            cand = tmp.copy()
                            break
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    for i in range(1, rotRank + 1):
                        if tooClose:
                            break
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # reflection
                        # across a plane that goes through axis Z and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    if tooClose:
                        numIons -= rotRank
                    else:
                        numIons -= 2 * rotRank
            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            if numIons == 0:
                break
    elif 'Ch' in nsym or 'ch' in nsym:  # Cnh group - Cn axis and horisontal mirror plane
        rotRank = int(nsym[2:])  # ceil(rand(1)*nsym);
        if rotRank > 2:
            newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                                   [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                                   [0.0, 0.0, lattice[2][2]]])
        candidate = []
        while 1:
            if numIons == 1:
                tmp = np.array([0.0, 0.0, 0.0])
                try:
                    candidate = np.vstack((candidate, tmp))
                except ValueError:
                    candidate = tmp.copy()
                break
            if numIons < 2 * rotRank:
                if numIons < rotRank:  # put atoms on axis
                    cand = np.array([[0.0, 0.0, np.random.random() - 0.5]])
                    while abs(cand[0][2]) < 0.01:
                        cand = np.array([[0.0, 0.0, np.random.random() - 0.5]])
                    cand = np.vstack((cand, np.dot(cand, Hz)))
                    numIons -= 2
                else:  # put atoms in mirror plane XOY
                    tmp = np.random.random((1, 3)) - 0.5
                    # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5] and then add (0.5,0.5,0.5)
                    while ellipse_mode and np.linalg.norm(np.array((tmp[0][0], tmp[0][1]))) > 0.5:
                        tmp = tmp = np.random.random((1, 3)) - 0.5
                    tmp[0][2] = 0
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    numIons -= rotRank
            else:
                tmp = np.random.random((1, 3)) - 0.5
                # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5] and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5
                if ((tmp[0][0] * newLattice[0][0]) ** 2 + (tmp[0][1] * newLattice[1][1]) ** 2 < minDistance ** 2) and \
                        (rotRank > 1):  # too close to the main axis
                    numIons -= 2
                    cand = np.array([[0.0, 0.0, tmp[0][2]], [0.0, 0.0, -tmp[0][2]]])
                else:
                    if abs(np.dot(tmp[0][2], newLattice[2][2])) < minDistance / 2.0:
                        tooClose = 1
                        tmp[0][2] = 0
                    else:
                        tooClose = 0
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    if tooClose == 0:
                        cand1 = np.dot(cand, Hz)
                        cand = np.vstack((cand, cand1))  # reflected atoms in the mirror plane
                    if tooClose:
                        numIons -= rotRank
                    else:
                        numIons -= 2 * rotRank

            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))

            if numIons == 0:
                break
    elif 'Dh' in nsym or 'dh' in nsym:  # dihedral symmetry with horisontal mirror plane XOY
        rotRank = int(nsym[2:])  # ceil(rand(1)*nsym);
        if rotRank > 2:
            newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                                   [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                                   [0.0, 0.0, lattice[2][2]]])
        candidate = []
        while 1:
            if numIons == 1:
                tmp = np.array([0.0, 0.0, 0.0])
                try:
                    candidate = np.vstack((candidate, tmp))
                except ValueError:
                    candidate = tmp.copy()
                break
            if numIons < 4 * rotRank:
                if numIons > 2 * rotRank:  # put atoms in the mirror plane, another possibility is to put them
                    # above/below the rotation axis (so that rotation = mirroring)
                    # (TODO: ALTERNATE OR CHOOSE METHOD RANDOMLY!)
                    tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                    # and then add (0.5,0.5,0.5)
                    while ellipse_mode and np.linalg.norm(np.array((tmp[0][0], tmp[0][1]))) > 0.5:
                        tmp = np.random.random((1, 3)) - 0.5
                    tmp[0][2] = 0
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    for i in range(1, rotRank + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, -1.0]])  # rotation
                        # by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    numIons -= 2 * rotRank
                elif numIons < rotRank:  # put atoms on axis
                    cand = np.array([[0, 0, np.random.random() - 0.5]])
                    cand = np.vstack((cand, np.dot(cand, Hz)))  # because of 2fold rotational axes and mirror plane
                    numIons -= 2
                else:  # put atoms on 2nd order rotation axes
                    tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5] and then
                    # add (0.5,0.5,0.5)
                    r = np.linalg.norm(tmp)
                    tmp = np.array([[r, 0.0, 0.0]])  # axe X
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    numIons -= rotRank
            else:
                tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                # and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5
                if ((tmp[0][0] * newLattice[0][0]) ** 2 + (tmp[0][1] * newLattice[1][1]) ** 2 < minDistance ** 2) and \
                        (rotRank > 1):  # too close to the main axis
                    numIons = numIons - 2
                    cand = np.array([[0.0, 0.0, tmp[0][2]],
                                     [0.0, 0.0, -tmp[0][2]]])
                else:
                    if abs(tmp[0][2] * newLattice[2][2]) < minDistance / 2.0:
                        tooClose1 = 1
                        tmp[0][2] = 0
                    else:
                        tooClose1 = 0
                    tooClose2 = 0
                    for i in range(1, rotRank + 1):  # check if atom is too close to some
                        # rotation axis (in this case - put atom on this axis)
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, -1.0]])  # rotation
                        # by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        if np.linalg.norm(np.dot((nextAtom - tmp), newLattice)) < minDistance:
                            tooClose2 = 1
                            tooClose1 = 1  # since average atom is in the mirror plane (mirror plane contains
                            # rotation axes axes)
                            tmp = (tmp + nextAtom) / 2.0
                            cand = tmp.copy()
                            break

                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    if tooClose2 == 0:
                        for i in range(1, rotRank + 1):
                            angle = 2 * i * np.pi / rotRank
                            opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                               [np.sin(angle), -np.cos(angle), 0.0],
                                               [0.0, 0.0, -1.0]])  #
                            # rotation by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                            nextAtom = np.dot(tmp, opemat)
                            cand = np.vstack((cand, nextAtom))
                    if tooClose1 == 0:
                        cand1 = np.dot(cand, Hz)
                        cand = np.vstack((cand, cand1))  # reflected atoms in the mirror plane

                    if tooClose2:
                        numIons -= rotRank
                    elif tooClose1:
                        numIons -= 2 * rotRank
                    else:
                        numIons -= 4 * rotRank

            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))

            if numIons == 0:
                break
    elif 'Dd' in nsym or 'dd' in nsym or 'Dv' in nsym or 'dv' in nsym:
        # dihedral symmetry with vertical mirror planes (Dnd, Dnv)
        rotRank = int(nsym[2:])  # ceil(rand(1)*nsym);
        if rotRank > 1:
            newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                                   [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                                   [0.0, 0.0, lattice[2][2]]])
        candidate = []
        while 1:
            if numIons == 1:
                tmp = np.array([0.0, 0.0, 0.0])
                try:
                    candidate = np.vstack((candidate, tmp))
                except ValueError:
                    candidate = tmp.copy()
                break
            if numIons < 4 * rotRank:
                if numIons < 2 * rotRank:  # put atoms on axis
                    cand = np.array([[0.0, 0.0, np.random.random() - 0.5]])
                    cand = np.vstack((cand, np.dot(cand, Hz)))  # because of 2fold rotational axes
                    numIons -= 2
                else:  # put atoms on 2nd order rotation axes, can also put on mirror planes
                    #  (toDo: ALTERNATE OR CHOOSE METHOD RANDOMLY!)
                    tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                    # and then add (0.5,0.5,0.5)
                    r = np.linalg.norm(tmp)
                    tmp = np.array([[r, 0.0, 0.0]])  # axe X
                    cand = tmp.copy()
                    angleM = np.pi / rotRank
                    opematM = np.array([[np.cos(angleM), np.sin(angleM), 0.0],
                                        [np.sin(angleM), -np.cos(angleM), 0.0],
                                        [0.0, 0.0, 1.0]])  # mirror plane
                    mirrorAtom = np.dot(tmp, opematM)
                    cand = np.vstack((cand, mirrorAtom))
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                        nextMirrorAtom = np.dot(mirrorAtom, opemat)
                        cand = np.vstack((cand, nextMirrorAtom))
                    numIons -= 2 * rotRank
            else:
                tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                # and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5
                if (tmp[0][0] * newLattice[0][0]) ** 2 + (tmp[0][1] * newLattice[1][1]) ** 2 < minDistance ** 2:
                    # too close to the main axis
                    numIons -= 1
                    cand = np.array([[0, 0, tmp[0][2]]])
                else:
                    cand = tmp.copy()
                    tooClose1 = 0
                    for i in range(1, rotRank + 1):  # check if atom is too close to some mirror plane
                        # (in this case - put atom on that plane)
                        angle = (2 * i * np.pi / rotRank) + np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # reflection
                        # across a plane that goes through axis Z and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        if np.linalg.norm(np.dot((nextAtom - tmp), newLattice)) < minDistance:
                            tooClose1 = 1
                            tmp = (tmp + nextAtom) / 2.0
                            cand = tmp.copy()
                            break
                    tooClose2 = 0
                    for i in range(1, rotRank + 1):  # check if atom is too close to some rotation axis
                        #  (in this case - put atom on this axis)
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, -1.0]])  # rotation
                        # by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        if np.linalg.norm(np.dot((nextAtom - tmp), newLattice)) < minDistance:
                            tooClose2 = 1
                            tmp = (tmp + nextAtom) / 2.0
                            cand = tmp.copy()
                            break

                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    if tooClose2 == 0:
                        for i in range(1, rotRank + 1):
                            angle = 2 * i * np.pi / rotRank
                            opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                               [np.sin(angle), -np.cos(angle), 0.0],
                                               [0.0, 0.0, -1.0]])  #
                            # rotation by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                            nextAtom = np.dot(tmp, opemat)
                            cand = np.vstack((cand, nextAtom))

                    if (tooClose2 == 1) or ((tooClose2 == 0) and (tooClose1 == 0)):
                        angleM = np.pi / rotRank
                        opematM = np.array([[np.cos(angleM), np.sin(angleM), 0.0],
                                            [np.sin(angleM), -np.cos(angleM), 0.0],
                                            [0.0, 0.0, 1.0]])  # mirror
                        # plane between two rotation axes (axis X being one of them)
                        mirrorAtom = np.dot(tmp, opematM)
                        cand = np.vstack((cand, mirrorAtom))
                        for i in range(1, rotRank - 1 + 1):
                            angle = 2 * i * np.pi / rotRank
                            opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                               [np.sin(angle), np.cos(angle), 0.0],
                                               [0.0, 0.0, 1.0]])  # rotate
                            # around Z axis
                            nextMirrorAtom = np.dot(mirrorAtom, opemat)
                            cand = np.vstack((cand, nextMirrorAtom))
                        if tooClose2 == 0:
                            for i in range(1, rotRank + 1):
                                angle = 2 * i * np.pi / rotRank
                                opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                                   [np.sin(angle), -np.cos(angle), 0.0],
                                                   [0.0, 0.0, -1.0]])  #
                                #  rotation by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                                nextMirrorAtom = np.dot(mirrorAtom, opemat)
                                cand = np.vstack((cand, nextMirrorAtom))

                    if tooClose2 == 1:  # atoms on axes
                        numIons -= 2 * rotRank
                    elif tooClose1 == 1:  # atoms on mirror planes
                        numIons -= 2 * rotRank
                    else:
                        numIons -= 4 * rotRank

            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))

            if numIons == 0:
                break
    elif 'D' in nsym or 'd' in nsym:  # dihedral symmetry
        rotRank = int(nsym[-1:])  # ceil(rand(1)*nsym);
        if rotRank > 2:
            newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                                   [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                                   [0.0, 0.0, lattice[2][2]]])
        candidate = []
        while 1:
            if numIons == 1:
                tmp = np.array([0.0, 0.0, 0.0])
                try:
                    candidate = np.vstack((candidate, tmp))
                except ValueError:
                    candidate = tmp.copy()
                break
            if numIons < 2 * rotRank:
                if numIons < rotRank:  # put atoms on axis
                    cand = np.array([[0, 0, np.random.random() - 0.5]])
                    cand = np.vstack((cand, np.dot(cand, Hz)))  # because of 2fold rotational axes
                    numIons -= 2
                else:  # put atoms on 2nd order rotation axes
                    tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                    # and then add (0.5,0.5,0.5)
                    r = np.linalg.norm(tmp)
                    tmp = np.array([[r, 0.0, 0.0]])  # axe X
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    numIons -= rotRank
            else:
                tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                # and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5
                if (tmp[0][0] * newLattice[0][0]) ** 2 + (tmp[0][1] * newLattice[1][1]) ** 2 < minDistance ** 2:  #
                    # too close to the main axis
                    numIons -= 1
                    cand = np.array([[0, 0, tmp[0][2]]])
                else:
                    cand = tmp.copy()
                    tooClose = 0
                    for i in range(1, rotRank + 1):  # check if atom is too close to some rotation axis
                        # (in this case - put atom on this axis)
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                           [np.sin(angle), -np.cos(angle), 0.0],
                                           [0.0, 0.0, -1.0]])  # rotation
                        # by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                        nextAtom = np.dot(tmp, opemat)
                        if np.linalg.norm(np.dot((nextAtom - tmp), newLattice)) < minDistance:
                            tooClose = 1
                            tmp = (tmp + nextAtom) / 2.0
                            cand = tmp.copy()
                            break

                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    if tooClose == 0:
                        for i in range(1, rotRank + 1):
                            angle = 2 * i * np.pi / rotRank
                            opemat = np.array([[np.cos(angle), np.sin(angle), 0.0],
                                               [np.sin(angle), -np.cos(angle), 0.0],
                                               [0.0, 0.0, -1.0]])  #
                            # rotation by Pi around axis that lies in XOY and makes an angle pi*i/n with the axis X
                            nextAtom = np.dot(tmp, opemat)
                            cand = np.vstack((cand, nextAtom))

                    if tooClose:
                        numIons -= rotRank
                    else:
                        numIons -= 2 * rotRank

            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))

            if numIons == 0:
                break

    elif 'O' in nsym or 'o' in nsym:  # O (cube rotational) and Oh (full cube) symmetries
        newLattice = np.diag((volLat ** (1 / 3.),) * 3)
        candidate = []
        if numIons != 1 and numIons != 6 and numIons != 7 and numIons != 8 and numIons != 9 and numIons < 12:
            status = 'Impossible to build the cluster with {0} atoms that has symmetry group {1}'.format(numIons, nsym)
            with open('error_cluster_symmetry', 'w') as f: f.write(status)
            errorS = 1
        if numIons % 2 == 1:  # put in the center
            candidate = np.array([0.0, 0.0, 0.0])
            numIons -= 1
        n24 = 0
        numIons1 = numIons
        while 1:
            numIons1 -= 24
            if numIons1 != 6 and numIons1 != 8 and numIons1 != 0 and numIons1 < 12:
                break
            else:
                n24 += 1
        n = 1
        while n <= n24:  # do a 'full' symmetry point
            tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
            # and then add (0.5,0.5,0.5)
            while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                tmp = np.random.random((1, 3)) - 0.5
            if ('Oh' in nsym or 'oh' in nsym) and n24 % 2 == 1 and n == n24:  # Oh (full cube) symmetry
                w = np.random.randint(1, 3)
                if w == 1:  # do only 24 poins :) (by generating a point on mirror plane containing some axis)
                    tmp[0][np.random.randint(0, 3)] = 0
                else:  # do only 24 poins, different way :)  (by generating a point on mirror plane containing some face diagonal)
                    w = np.random.randint(1, 4)
                    if w == 1:
                        tmp[0][0] = tmp[0][1]
                    elif w == 2:
                        tmp[0][0] = tmp[0][2]
                    else:
                        tmp[0][2] = tmp[0][1]
            cand = tmp.copy()
            # rotations around X, Y, Z (O is in the cube center, Z goes up)
            for i in range(1, 3 + 1):
                angle = i * np.pi / 2
                # opemat = [cos(angle) -sin(angle) 0; sin(angle) cos(angle) 0; 0 0 1] 
                opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                   [np.sin(angle), np.cos(angle), 0.0],
                                   [0.0, 0.0, 1.0]])  # rotate around
                # Z axis
                nextAtom = np.dot(tmp, opemat)
                cand = np.vstack((cand, nextAtom))
            for i in range(1, 3 + 1):
                angle = i * np.pi / 2
                # opemat = [1 0 0; 0 cos(angle) -sin(angle); 0 sin(angle) cos(angle)]; 
                opemat = np.array([[1.0, 0.0, 0.0],
                                   [0.0, np.cos(angle), - np.sin(angle)],
                                   [0.0, np.sin(angle), np.cos(angle)]])  # rotate around X axis
                nextAtom = np.dot(tmp, opemat)
                cand = np.vstack((cand, nextAtom))
            for i in range(1, 3 + 1):
                angle = i * np.pi / 2
                # opemat = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 cos(angle)];
                opemat = np.array([[np.cos(angle), 0.0, np.sin(angle)],
                                   [0.0, 1.0, 0.0],
                                   [- np.sin(angle), 0.0, np.cos(angle)]])  # rotate around Y axis
                nextAtom = np.dot(tmp, opemat)
                cand = np.vstack((cand, nextAtom))
            # rotations around 2-fold axes in XOY plane (connecting the middle of cube edges)
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, 0.0, -1.0]]))
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, 0.0, -1.0]]))
            cand = np.vstack((cand, nextAtom))
            # rotations around 2-fold axes in YOZ plane (connecting the middle of cube edges)
            nextAtom = np.dot(tmp, np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [0.0, 1.0, 0.0]]))
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[-1.0, 0.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [0.0, -1.0, 0.0]]))
            cand = np.vstack((cand, nextAtom))
            # rotations around 2-fold axes in ZOX plane (connecting the middle of cube edges)
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [0.0, -1.0, 0.0],
                                             [1.0, 0.0, 0.0]]))
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [0.0, -1.0, 0.0],
                                             [-1.0, 0.0, 0.0]]))
            cand = np.vstack((cand, nextAtom))
            # rotations around the main cube diagonals (3-fold axes)
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0]]))  # +++ diagonal, -120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))  # -240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 1.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [-1.0, 0.0, 0.0]]))  # ++- diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [0.0, 0.0, 1.0],
                                             [-1.0, 0.0, 0.0]]))  # +-- diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, -1.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, -1.0, 0.0],
                                             [0.0, 0.0, -1.0],
                                             [1.0, 0.0, 0.0]]))  # +-+ diagonal, 120 degree
            cand = np.vstack((cand, nextAtom))
            nextAtom = np.dot(tmp, np.array([[0.0, 0.0, 1.0],
                                             [-1.0, 0.0, 0.0],
                                             [0.0, -1.0, 0.0]]))  # 240 degree
            cand = np.vstack((cand, nextAtom))
            if ('Oh' in nsym or 'oh' in nsym) and not (n24 % 2 == 1 and n == n24):  # Oh (full cube) symmetry
                numIons -= 48
                n += 2
                cand1 = -1 * cand
                cand = np.vstack((cand, cand1))  # added inversion
            else:
                numIons -= 24
                n += 1
            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))
            if n > n24:
                break
        if numIons > 0:  # numIons can be described as 6x+8y
            n1 = np.ceil(numIons / 8.).astype(int)
            n2 = np.floor(numIons / 6.).astype(int)
            n = n1 + (np.round(np.random.random() * (n2 - n1))).astype(
                int)  # numIons = (4n-numIons/2)*6 + (numIons/2-3n)*8
            x = 4 * n - (np.round(numIons / 2.)).astype(int)
            y = np.round(numIons / 2.).astype(int) - 3 * n
            for i in range(1, x + 1):  # 6 atoms per go
                tmp = np.random.random() - 0.5
                while abs(tmp) < 0.05:
                    tmp = np.random.random() - 0.5
                if len(candidate) == 0:
                    candidate = np.array([[0.0, 0.0, tmp]])
                else:
                    candidate = np.vstack((candidate, np.array([[0.0, 0.0, tmp]])))
                candidate = np.vstack((candidate, np.array([[0.0, 0.0, -tmp]])))
                candidate = np.vstack((candidate, np.array([[0.0, tmp, 0.0]])))
                candidate = np.vstack((candidate, np.array([[0.0, -tmp, 0.0]])))
                candidate = np.vstack((candidate, np.array([[tmp, 0.0, 0.0]])))
                candidate = np.vstack((candidate, np.array([[-tmp, 0.0, 0.0]])))
            for i in range(1, y + 1):  # 8 atoms per go
                tmp = np.random.random() - 0.5
                while abs(tmp) < 0.05 or \
                        ((ellipse_mode == 1) and np.linalg.norm(np.array([[tmp, tmp, tmp]])) > 0.5):
                    tmp = np.random.random() - 0.5

                if len(candidate) == 0:
                    candidate = np.array([[tmp, tmp, tmp]])
                else:
                    candidate = np.vstack((candidate, np.array([[tmp, tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, -tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[tmp, -tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, tmp, -tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, -tmp, tmp]])))
                candidate = np.vstack((candidate, np.array([[-tmp, -tmp, -tmp]])))
    else:  # Cn symmetry
        rotRank = int(nsym[1:])  # ceil(rand(1)*nsym);
        if rotRank > 2:
            # newLattice = [sqrt(lattice(1,1)*lattice(2,2)) 0 0; 0 sqrt(lattice(1,1)*lattice(2,2)) 0; 0 0 lattice(3,3)]
            newLattice = np.array([[np.sqrt(lattice[0][0] * lattice[1][1]), 0.0, 0.0],
                                   [0.0, np.sqrt(lattice[0][0] * lattice[1][1]), 0.0],
                                   [0.0, 0.0, lattice[2][2]]])
        candidate = []
        while 1:
            if numIons < rotRank:  # put the rest of the atoms on the axis
                cand = np.array([[0, 0, np.random.random() - 0.5]])
                for i in range(2, numIons + 1):
                    tmp = np.array([[0, 0, np.random.random() - 0.5]])  # we work in the space
                    # [-0,5:0.5;-0,5:0.5;-0,5:0.5] and then add (0.5,0.5,0.5)
                    cand = np.vstack((cand, tmp))
                numIons = 0
            else:
                tmp = np.random.random((1, 3)) - 0.5  # we work in the space [-0,5:0.5;-0,5:0.5;-0,5:0.5]
                # and then add (0.5,0.5,0.5)
                while ellipse_mode and np.linalg.norm(tmp) > 0.5:
                    tmp = np.random.random((1, 3)) - 0.5
                if (tmp[0][0] * newLattice[0][0]) ** 2 + (tmp[0][1] * newLattice[1][1]) ** 2 < minDistance ** 2:
                    # too close to the main axis
                    numIons -= 1
                    cand = np.array([[0.0, 0.0, tmp[0][2]]])
                else:
                    cand = tmp.copy()
                    for i in range(1, rotRank - 1 + 1):
                        angle = 2 * i * np.pi / rotRank
                        opemat = np.array([[np.cos(angle), - np.sin(angle), 0.0],
                                           [np.sin(angle), np.cos(angle), 0.0],
                                           [0.0, 0.0, 1.0]])  # rotate
                        # around Z axis
                        nextAtom = np.dot(tmp, opemat)
                        cand = np.vstack((cand, nextAtom))
                    numIons -= rotRank

            if len(candidate) == 0:
                candidate = cand.copy()
            else:
                candidate = np.vstack((candidate, cand))

            if numIons == 0:
                break

    if len(candidate) == 0:
        errorS = 1

    AbsoluteCoord = np.dot(candidate, newLattice)
    b, a = type(rand_cell).PrincipleAxis(AbsoluteCoord)  # find the principle rotation axis

    if 'E' in nsym:
        AbsoluteCoord = np.dot(AbsoluteCoord, a)  # maximal moment of inertia - for Z axis, minimum - for X
        # we will fix the positions so that no atom is outside of the unit cell and cluster basically fits the whole cell
        #  mass_center = zeros(1,3);
        #  for i = 1 : 3
        #   mass_center(i) = sum(AbsoluteCoord(:,i));
        #   AbsoluteCoord(:,i) = AbsoluteCoord(:,i) - mass_center(i); % 'center' the cluster, center of mass should be exactly at [0.5;0.5;0.5]
        #  end
        ma = np.zeros((3), dtype=float)
        mi = np.zeros((3), dtype=float)
        for i in range(3): ma[i] = AbsoluteCoord[:, i].max()
        for i in range(3): mi[i] = AbsoluteCoord[:, i].min()
        newLattice = np.diag((ma[0] - mi[0] + 0.02, ma[1] - mi[1] + 0.02, ma[2] - mi[2] + 0.02))
        AbsoluteCoord[:, 0] = AbsoluteCoord[:, 0] - mi[0] + 0.01  # 'center' the cluster
        AbsoluteCoord[:, 1] = AbsoluteCoord[:, 1] - mi[1] + 0.01
        AbsoluteCoord[:, 2] = AbsoluteCoord[:, 2] - mi[2] + 0.01
        candidate = np.dot(AbsoluteCoord, np.linalg.inv(newLattice)) - 0.5
    elif 'O' in nsym or 'o' in nsym or 'T' in nsym or 't' in nsym:  # cubic lattice
        candidate = np.dot(candidate, a)  # maximal moment of inertia - for Z axis, minimum - for X
        # we will fix the positions so that no atom is outside of the unit cell and cluster basically fits the whole cell
        m = np.zeros((3))
        for i in range(3):
            m[i] = np.absolute(candidate[:, i]).max()  # m = max(max(abs(candidate)))
            candidate[:, i] = candidate[:, i] * (0.5 / (m[i] + 0.01))
    else:  # 'cylindrical' lattice and inversion
        # candidate = candidate*a;
        # lat = inv(a)*lat*a; # this way AbsoluteCoord => AbsoluteCoord*a (we rotate lattice and relative coordinates)
        AbsoluteCoord = np.dot(AbsoluteCoord, a)  # maximal moment of inertia - for Z axis, minimum - for X
        mass_center = np.zeros((3), dtype=float)
        for i in range(3):
            mass_center[i] = AbsoluteCoord[:, i].sum() / nI
            AbsoluteCoord[:, i] = AbsoluteCoord[:, i] - mass_center[i]  # 'center' the cluster, center of mass
            # should be exactly at [0.5;0.5;0.5]

        # we will fix the positions so that no atom is outside of the unit cell and cluster basically fits the whole cell
        m = np.zeros((3))
        for i in range(3): m[i] = np.absolute(AbsoluteCoord[:, i]).max()
        newLattice = np.diag((2 * m[0] + 0.02, 2 * m[1] + 0.02, 2 * m[2] + 0.02))
        # This implements MATLAB candidate = (AbsoluteCoord/newLattice)
        candidate = np.dot(AbsoluteCoord, np.linalg.inv(newLattice))

    candidate += 0.5

    if errorS == 0:
        return candidate, newLattice
    else:
        raise RuntimeError("Symope failed.")


"""
"""

if __name__ == "__main__":
    nsym = 'Th'
    numIons = 36
    lat = np.diag((4.9714, 4.9714, 4.9714))
    minDistMatrix = 1.09
    # minDistMatrix = np.array(np.mtrx('0.8 0.6; 0.6 0.8'))
    candidate, lat, errorS = symope_cluster(nsym, numIons, lat, minDistMatrix)
    print(candidate)
