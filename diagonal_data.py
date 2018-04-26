import numpy as np

a = np.array([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
#print a.flat[:]
#print a.flat[0]
#print len(a)
"""
b = 0
for i in range(0, len(a)):
    print a.flat[b]
    b += len(a)+1
"""
from sklearn import datasets

digits = datasets.load_digits()
print(digits.target)
