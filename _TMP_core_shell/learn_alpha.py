import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import alphashape
from USPEX.Atomistic.Transformation import Transformation
from USPEX.components import AtomisticRepresentation
from USPEX.components import AtomicStructure

matplotlib.use('TkAgg')
core_structure = AtomisticRepresentation.readXYZ('Au-51-15.xyz')
core_coords = core_structure.getCartesianCoordinates()
adsorbant_structure_original = AtomisticRepresentation.readXYZ('phenyl2.xyz')
mount_point = np.array([1.21940,  -0.16520,   2.16000])
orientation = np.array([0.68250,  -0.09240,   1.20870]) - mount_point
orientation /= np.linalg.norm(orientation)

alpha_shape = alphashape.alphashape(core_coords, 0.1)

vert_normals = []
for v, vertice in enumerate(alpha_shape.vertices):
    normal = np.array([0., 0., 0.])
    neighbors = []
    for f, face in enumerate(alpha_shape.faces):
        if len(np.intersect1d([v], face)) == 1:
            neighbors.append(f)
            normal += alpha_shape.face_normals[f]
    print(neighbors, ' vs ', np.sort(alpha_shape.vertex_faces[v]))
    rc = vertice
    normal /= np.linalg.norm(normal)
    vert_normals.append({'mount_point': rc, 'orientation': normal})
normals = vert_normals
normals_2 = alpha_shape.vertex_normals

res_structure = core_structure
for k in range(5):
    random_angle = 2 * np.pi * np.random.random()
    random_rotation = Transformation.fromRotVector(orientation * random_angle, -mount_point)
    adsorbant_structure = random_rotation.transform(adsorbant_structure_original)

    chosen_normal = np.random.choice(normals)
    rot_ax = np.cross(orientation, chosen_normal['orientation'])
    rot_ax /= np.linalg.norm(rot_ax)
    alpha = np.arccos(orientation @ chosen_normal['orientation'])
    rotation = Transformation.fromRotVector(alpha * rot_ax, chosen_normal['mount_point'])
    adsorbant_structure = rotation.transform(adsorbant_structure)

    atomTypes = []
    coordinates = []
    for molecule in [adsorbant_structure, res_structure]:
        atomTypes.extend(molecule.getAtomTypes())
        coordinates.extend(molecule.getCartesianCoordinates())
    res_structure = AtomicStructure(atomTypes, coordinates, None)

AtomisticRepresentation.writeXYZ('res.xyz', res_structure)
# res_atoms.write('attached.xyz', format='xyz')

# 1. centers and normals of faces
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.scatter(*core_coords.T)
ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.6)

for i in range(len(alpha_shape.faces)):
    r_cm = alpha_shape.triangles_center[i]
    normal = alpha_shape.face_normals[i]

    ax.scatter(*r_cm, facecolors='g', edgecolors='g')
    ax.scatter(*(r_cm + normal), facecolors='r', edgecolors='r')
    ax.plot(*np.vstack((r_cm, r_cm + normal)).T)

# 2. edges
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.scatter(*core_coords.T)
ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.2)
core_coords = alpha_shape.vertices


for edge in alpha_shape.edges:
    f_adj = []
    for f, face in enumerate(alpha_shape.faces):
        if len(np.intersect1d(edge, face)) == 2:
            f_adj.append(f)

    print(f_adj)

    rc = 0.5 * (core_coords[edge[0]] + core_coords[edge[1]])
    normal = alpha_shape.face_normals[f_adj[0]] + alpha_shape.face_normals[f_adj[1]]
    normal /= np.linalg.norm(normal)

    ax.scatter(*rc, facecolors='g', edgecolors='g')
    ax.scatter(*(rc + normal), facecolors='r', edgecolors='r')
    ax.plot(*np.vstack((rc, rc + normal)).T)

# 3. vertices
core_coords = core_structure.getCartesianCoordinates()
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.scatter(*core_coords.T)
ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.2)

for v, vertice in enumerate(alpha_shape.vertices):
    normal = np.array([0., 0., 0.])
    for f, face in enumerate(alpha_shape.faces):
        if len(np.intersect1d([v], face)) == 1:
            normal += alpha_shape.face_normals[f]

    rc = vertice
    # normal = alpha_shape.face_normals[f_adj[0]] + alpha_shape.face_normals[f_adj[1]] + alpha_shape.face_normals[f_adj[2]]
    normal /= np.linalg.norm(normal)

    ax.scatter(*rc, facecolors='g', edgecolors='g')
    ax.scatter(*(rc + normal), facecolors='r', edgecolors='r')
    ax.plot(*np.vstack((rc, rc + normal)).T)
matplotlib.use('TkAgg')
plt.show()