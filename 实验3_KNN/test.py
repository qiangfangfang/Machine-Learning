import numpy as np

dataSet = np.array(
        [[1.0, 1.1],
        [1.0, 1.0],
        [  0,   0],
        [  0, 0.1]]
)
dataSetsize = dataSet.shape[0]
print(dataSetsize)
inx = [1.2, 1]
A = np.tile(inx, (dataSetsize, 1))
print(A)