# MLSDeformation

This is an example of the MLS affine transform function featured in https://people.engr.tamu.edu/schaefer/research/mls.pdf
This function will take in an array v, p, q, where v is the points of the mesh to be transformed, p is the initial landmark points before transform,
and q is the landmark points after the transform. The function will take in these values and return a transformed list of coordinates of v.
Note that p and q will have the same dimension and the returned array will have the same dimension as input v.
