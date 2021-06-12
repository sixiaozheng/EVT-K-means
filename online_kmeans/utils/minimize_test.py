from scipy.stats import genextreme
import numpy as np 
from scipy.optimize import minimize

rvs = genextreme.rvs(c=0.5, loc=1, scale=0.8, size=1000)
print(rvs)
res = genextreme.fit(rvs)
print(res)

def gev_lf(param, x):
    res = np.sum(np.log(genextreme.pdf(x,c=param[0], loc=param[1], scale=param[2])))
    return res

param = np.array([0.5,1,1])

res= minimize(gev_lf, param, args =rvs, method='nelder-mead',options={'xatol':1e-8,'disp':True})
print(res.x)
