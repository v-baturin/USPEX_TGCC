import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import alphashape
from ase.io import read
# from USPEX.Atomistic.Transformation import Transformation
# from USPEX.components import AtomisticRepresentation
# from USPEX.components import AtomicStructure

core_structure = read('Au-51-15.xyz')
# points_3d = core_structure.positions
points_3d = 10*np.array([
    (0., 0., 0.), (0., 0., 1.), (0., 1., 0.),
    (1., 0., 0.), (1., 1., 0.), (1., 0., 1.),
    (0., 1., 1.), (1., 1., 1.), (.25, .5, .5),
    (.5, .25, .5), (.5, .5, .25), (.75, .5, .5),
    (.5, .75, .5), (.5, .5, .75)
])

alpha_shape = alphashape.alphashape(points_3d, 0.21)
# alpha_shape.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.scatter(*points_3d.T)
ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces, alpha=0.1)

## 1. Faces
# for i in range(len(alpha_shape.faces)):
#     r_cm = alpha_shape.triangles_center[i]
#     normal = alpha_shape.face_normals[i]
#
#     ax.scatter(*r_cm, facecolors='g', edgecolors='g')
#     ax.scatter(*(r_cm + normal), facecolors='r', edgecolors='r')
#     ax.plot(*np.vstack((r_cm, r_cm + normal)).T, 'r')

## 2. Edges
for adj_e, adj_f in zip(alpha_shape.face_adjacency_edges, alpha_shape.face_adjacency):

    rc = 0.5 * (alpha_shape.vertices[adj_e[0]] + alpha_shape.vertices[adj_e[1]])
    normal = alpha_shape.face_normals[adj_f[0]] + alpha_shape.face_normals[adj_f[1]]
    normal /= np.linalg.norm(normal)

    ax.scatter(*rc, facecolors='g', edgecolors='g')
    ax.scatter(*(rc + normal), facecolors='r', edgecolors='r')
    ax.plot(*np.vstack((rc, rc + normal)).T, 'r')



# ## 3. Vertices
# for v, vertice in enumerate(alpha_shape.vertices):
#     normal = np.array([0., 0., 0.])
#     for f, face in enumerate(alpha_shape.faces):
#         if len(np.intersect1d([v], face)) == 1:
#             normal += alpha_shape.face_normals[f]
#     builtin_normal = alpha_shape.vertex_normals[v]
#     rc = vertice
#     # normal = alpha_shape.face_normals[f_adj[0]] + alpha_shape.face_normals[f_adj[1]] + alpha_shape.face_normals[f_adj[2]]
#     normal /= np.linalg.norm(normal)
#
#     ax.scatter(*rc, facecolors='g', edgecolors='g')
#     ax.scatter(*(rc + normal), facecolors='r', edgecolors='r')
#     ax.plot(*np.vstack((rc, rc + normal)).T, 'b')
#     ax.plot(*np.vstack((rc, rc + builtin_normal)).T, 'g')





plt.show()