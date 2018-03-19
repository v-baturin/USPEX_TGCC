import numpy as np

def zmatrix2coord(zmatrix, format):

    #This function is to transfor Zmatrix to XYZ
    #Remember that Zmatrix is a define in spherical coordinates systems
    #So we need a lot of transformation from (r,theta,phi) to (x,y,z)

    N_atom  = len(zmatrix)
    coords  = np.zeros((N_atom, 3))
    origin  = zmatrix[0,:]
    if N_atom > 1:
        coords[1,2] = zmatrix[1,0]*np.cos(zmatrix[1,1])
        coords[1,0] = zmatrix[1,0]*np.sin(zmatrix[1,1])*np.cos(zmatrix[1,2])
        coords[1,1] = zmatrix[1,0]*np.sin(zmatrix[1,1])*np.sin(zmatrix[1,2])
        if N_atom > 2:
            for i in range(2,N_atom):
                if i==2:
                    ref = coords[format[2,:2] - 1,:]
                else:
                    ref = coords[format[i,:] - 1,:]
                coords[i, :]= GetXYZ(ref, zmatrix[i,:])
    coords += origin
    return coords

def GetXYZ(ref, zmatrix):
    #from reference (i = 1, 2, 3), we construct the followings
    #1, origin (1)
    #2, z-axis along 1->2
    #3, y-axis is perpendicular to plane (1-2-3)
    #the new sperical system as (r, theta, phi)
    r     = zmatrix[0]
    theta = zmatrix[1]
    phi   = -zmatrix[2]

    coor = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)])
    u1 = ref[1,:] - ref[0,:]
    if len(ref) == 2:
        u2 = np.array([1, 0, 0])
    else:
        u2 = ref[2,:] - ref[1,:]
    z = u1/np.linalg.norm(u1)
    y  = np.cross(u1,u2)
    y  = y/np.linalg.norm(y)
    x  = np.cross(y,z)
    x  = x/np.linalg.norm(x)
    #coor = coor/(np.stack([x,y,z]).T)
    coor = np.linalg.lstsq(np.stack([x,y,z]), coor)[0]
    return coor + ref[0,:]

