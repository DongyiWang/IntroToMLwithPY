# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as ip
from scipy import sparse
import mglearn
x = np.array([[1,2,3],[4,5,6]])
print("x:\n{}".format(x))
eye = np.eye(4)
print("Numpy array:\n{}".format(eye))
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x, y, marker="x")
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age' : [24, 13, 53, 33]
        }
data_pandas = pd.DataFrame(data)
ip.display(data_pandas)
ip.display(data_pandas[data_pandas.Age > 30])

import sys
print("Python version: {}".format(sys.version))
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))

