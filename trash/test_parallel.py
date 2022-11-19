from math import sqrt
from joblib import Parallel, delayed
import pdb

def test_return(i):
    return [i, i*2]

# tmp = Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
tmp = Parallel(n_jobs=1)(delayed(test_return)(i) for i in range(10))
pdb.set_trace()
print(tmp)
