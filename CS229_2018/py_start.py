import numpy as np
a = np.array([1,2,3])
print(a)
print(a.T) # for a vector, .T operation will not change it to a column vector, so we must use reshape() function
print(a.reshape(-1,1))
print(np.matmul(a,a.reshape(-1,1)))
# print(np.matmul(a.reshape(-1,1),a)) error
print(a.reshape(-1,1).T)