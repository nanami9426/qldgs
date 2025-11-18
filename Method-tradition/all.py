import numpy as np
from Function import Fun

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    max_iter = opts['T']
    dim = np.size(xtrain, 1)
    sf = np.ones((1, dim))
    num_feat   = dim
    x = np.ones([1, num_feat])
    fit = Fun(xtrain, xvalid, ytrain, yvalid, x[0], opts)
    curve = np.ones([1, max_iter]) * fit
    all_data = {'sf': sf, 'c': curve, 'nf': num_feat}
    return all_data