import numpy as np
import matplotlib.pyplot as plt
 
def multiGaussian(x, c, loc, scale):
    x = (x-loc)/scale
    if c == 0:
        y = (1/scale)*np.exp(-x)
    else:
        y = (1/scale)*np.power(1 + c * x, -1 - 1/c)
        y[(c>0)&(x<0)] = 0.0
        y[(c<0)&((x<0)|(x>-1/c))] = 0.0
    return y
 
def computeGamma(x,c, loc, scale,alpha,multiGaussian):
    n_samples=x.shape[0]
    n_clusters=len(alpha)
    gamma=np.zeros((n_samples,n_clusters))
    p=np.zeros(n_clusters)
    g=np.zeros(n_clusters)
    for i in range(n_samples):
        for j in range(n_clusters):
            p[j]=multiGaussian(x[i],c[j],loc[j],scale[j])
            g[j]=alpha[j]*p[j]
        for k in range(n_clusters):
            gamma[i,k]=g[k]/np.sum(g)
    return gamma
 
class MyGMM():
    def __init__(self,n_clusters,ITER=50):
        self.n_clusters=n_clusters
        self.ITER=ITER
        self.mu=0
        self.sigma=0
        self.alpha=0
      
    def fit(self,data):
        n_samples=data.shape[0]
        n_features=data.shape[1]
        '''
        mu=data[np.random.choice(range(n_samples),self.n_clusters)]
        '''
        alpha=np.ones(self.n_clusters)/self.n_clusters
        
        mu=np.array([[.403,.237],[.714,.346],[.532,.472]])
        
        sigma=np.full((self.n_clusters,n_features,n_features),np.diag(np.full(n_features,0.1)))
        for i in range(self.ITER):
            gamma=computeGamma(data,mu,sigma,alpha,multiGaussian)
            alpha=np.sum(gamma,axis=0)/n_samples
            for i in range(self.n_clusters):
                mu[i]=np.sum(data*gamma[:,i].reshape((n_samples,1)),axis=0)/np.sum(gamma,axis=0)[i]
                sigma[i]=0
                for j in range(n_samples):
                    sigma[i]+=(data[j].reshape((1,n_features))-mu[i]).T.dot((data[j]-mu[i]).reshape((1,n_features)))*gamma[j,i]
                sigma[i]=sigma[i]/np.sum(gamma,axis=0)[i]
        self.mu=mu
        self.sigma=sigma
        self.alpha=alpha
        
    def predict(self,data):
        pred=computeGamma(data,self.mu,self.sigma,self.alpha,multiGaussian)
        cluster_results=np.argmax(pred,axis=1)
        return cluster_results
