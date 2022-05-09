# meshA_to_meshB
 
A flexible linear interpolator which can interpolate data between any two sets of
structured or unstructured points (in 2D). It also has the ability to extrapolate
data on points outside the convex hull of input points (using nearest extrapolation).
The main benefit to this interpolator is that it is significantly faster than Scipy's griddata()
module for when interpolating data between the same two sets of points multiple times (for example for every time-step of a simulation).
Particularly, it scales much better as the number of points increases.

Thanks to Jaime on stackoverflow for the excellent linear interpolator function.
https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
