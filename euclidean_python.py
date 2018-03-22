import numpy as np
from numpy import linalg as LA
from scipy import spatial 

a = np.arange(9) - 4
print a

b = a.reshape((3, 3))
print b

print LA.norm(a)

print LA.norm(b)

print LA.norm(b, 'fro')

#dist = np.linalg.norm(a-b)

#print dist


jarak = spatial.distance.euclidean(10,12)
print "jarak : ", -jarak

a = [2, 4, 6, 8]

for i in a:
	print "jarak : ", spatial.distance.euclidean(i, 4)

