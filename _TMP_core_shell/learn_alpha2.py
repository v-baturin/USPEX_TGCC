import os
import sys
import numpy as np
from ase.io import read, write
from ase import Atoms
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape
from USPEX.Atomistic.Transformation import Transformation

# core_coords = 10*np.array([
#     (0., 0., 0.), (0., 0., 1.), (0., 1., 0.),
#     (1., 0., 0.), (1., 1., 0.), (1., 0., 1.),
#     (0., 1., 1.), (1., 1., 1.), (.25, .5, .5),
#     (.5, .25, .5), (.5, .5, .25), (.75, .5, .5),
#     (.5, .75, .5), (.5, .5, .75)
# ])

atoms_core = read('Au-51-15.xyz')
atoms_adsorbant = read('benzene.xyz')
mount_point = np.array([1.21940,  -0.16520,   2.16000])
axis = np.array([0.68250,  -0.09240,   1.20870]) - mount_point
axis /= np.linalg.norm(axis)
core_coords = atoms_core.positions
# adsorbant_coords = atoms_adsorbant.positions - mount_point
#
# random_angle = 2 * np.pi * np.random.random()
# random_rotation = Transformation.fromRotVector(axis * random_angle, 0)
# random_rot_displaced_ads = random_rotation.transformCoordinates(adsorbant_coords)
# atoms_adsorbant.positions = random_rot_displaced_ads
# atoms_adsorbant.write('step1.xyz')


alpha_shape = alphashape.alphashape(core_coords, 0.1)
points_3d = alpha_shape.vertices
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.scatter(*points_3d.T)
ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.1)

for i in range(len(alpha_shape.faces)):
    r_cm = alpha_shape.triangles_center[i]
    normal = alpha_shape.face_normals[i]

    ax.scatter(*r_cm, facecolors='g', edgecolors='g')
    ax.scatter(*(r_cm + normal), facecolors='r', edgecolors='r')
    ax.plot(*np.vstack((r_cm, r_cm + normal)).T)

plt.show()