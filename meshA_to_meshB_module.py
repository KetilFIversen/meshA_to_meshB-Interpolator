"""
@author: Ketil
"""

import numpy as np
from scipy.spatial     import Delaunay, cKDTree



# New Interpolator class
class meshA_to_meshB:
    """
    NEW 11.04.22
    meshA_to_meshB object which can be used to interpolate values defined on a set 
    of points defined on meshA to a set of points defined on meshB.
    
    This interpolator was based on the methods and ideas introduced in:
        *https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
        *https://stackoverflow.com/questions/51858194/storing-the-weights-used-by-scipy-griddata-for-re-use
        *https://programtalk.com/python-examples/scipy.interpolate.griddata.ravel/

    Parameters
    ----------
    meshA_points : array ([n_points,2])
                   Array of cellcenter points from meshA
    meshB_points : array ([n_points,2])
                   Array of cellcenter points from meshB
    extrap_qhull : bool
                   Whether or not to force nearest extrapolation on points lying on the convex hull.

    Returns
    -------
    meshA_to_meshB object
    
    Methods
    ----------
    interp_weights(meshA_points, meshB_points, d=2):
        Compute the necessary triangularization, vertices and weights. 
        This method is called only once during initialization of the meshA_to_meshB object.
    
    extrapolate_nans(mesh_points, data):
        Extrapolate any NANs using nearest neighbour interpolation which might arrise from linear interpolation.
        
    interpolate(data, extrap_nans=True):
        Interpolate data (defined on meshA) onto points on meshB using linear interpolation. 
        It is higly recommended to keep extrap_nans=True.
        If extrap_nans=True, then .extrapolate_nans() is automatically called to patch any NANs which may arrise.

    interpolate_nearest(data):
        Interpolate data (defined on meshA) onto points on meshB using nearest neighbor interpolation.

    """
    
    def __init__(self, meshA_points, meshB_points, extrap_qhull=True):
        
        self.meshA_points = meshA_points
        self.meshB_points = meshB_points
        self.extrap_qhull = extrap_qhull
        
        self.tri, self.vtx, self.wts = self.interp_weights(self.meshA_points, self.meshB_points)
        
        self.treeA = None
        self.treeB = None
        
        self.nansB    = None
        self.notnansB = None
        self.notnansA = None

        self.nansDetected = False


    def nearest_value(self, tree, values, xi):
            _, i = tree.query(xi, workers=-1)
            return values[i]
    
    
    # Extrapolate missing NAN values using nearest method
    def extrapolate_nans(self, meshA_points, meshB_points, dataA, dataB, extrap_from_A=True):
        if self.nansB is None or self.notnansB is None or self.notnansA is None:
            self.nansB        = np.isnan(dataB).nonzero()[0]
            self.notnansB     = np.logical_not(np.isnan(dataB)).nonzero()[0]
            self.notnansA     = np.logical_not(np.isnan(dataA)).nonzero()[0]
            self.xB, self.yB  = meshB_points[self.nansB,0], meshB_points[self.nansB,1]
            self.nansDetected = True
        if extrap_from_A:
            if self.treeA is None:
                self.treeA    = cKDTree(meshA_points[self.notnansA])
            dataB[self.nansB] = self.nearest_value(tree=self.treeA, values=dataA[self.notnansA], 
                                                   xi=np.transpose([self.xB, self.yB]))
        else:
            if self.treeB is None:
                self.treeB    = cKDTree(meshB_points[self.notnansB])
            dataB[self.nansB] = self.nearest_value(tree=self.treeB, values=dataB[self.notnansB], 
                                                   xi=np.transpose([self.xB, self.yB]))
        return dataB


    # Compute necessary triangularization, vertices and weights
    def interp_weights(self, meshA_points, meshB_points, d=2):
        triA = Delaunay(meshA_points)
        triB = Delaunay(meshB_points)
        simplex   = triA.find_simplex(meshB_points)
        vertices  = np.take(triA.simplices, simplex, axis=0)
        temp      = np.take(triA.transform, simplex, axis=0)
        delta     = meshB_points - temp[:, d]
        bary      = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        weights   = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        
        if self.extrap_qhull:
            convhull  = np.unique(triB.convex_hull.flatten()) 
            weights[convhull] = -1 # Force boundary points to be undefined, so that we can use nearest interpolation on them
        return triA, vertices, weights


    # Interpolate data from meshA to meshB using linear interpolation
    def interpolate(self, data, extrap_nans=True, extrap_from_A=True, fill_value=0.0):
        data_interp = np.einsum('nj,nj->n', np.take(data, self.vtx), self.wts)
        data_interp[np.any(self.wts < 0, axis=1)] = np.nan 
        if extrap_nans and (self.nansDetected or np.any(np.isnan(data_interp))):
            data_interp = self.extrapolate_nans(meshA_points=self.meshA_points, 
                                                meshB_points=self.meshB_points, 
                                                dataA=data, dataB=data_interp,
                                                extrap_from_A=extrap_from_A)
        else:
            data_interp[np.any(self.wts < 0, axis=1)] = fill_value # Use fill value as FiPy will crash if NANs are encountered.
        return data_interp
    
    # Interpolate data from meshA to meshB using nearest interpolation
    def interpolate_nearest(self, data):
        if self.treeA is None or self.notnansA is None:
            self.notnansA = np.logical_not(np.isnan(data)).nonzero()[0]
            self.treeA    = cKDTree(self.meshA_points[self.notnansA])
        self.xB, self.yB = self.meshB_points[:,0], self.meshB_points[:,1]
        data_interp = self.nearest_value(tree=self.treeA, values=data[self.notnansA], 
                                                                 xi=np.transpose([self.xB, self.yB]))
        return data_interp