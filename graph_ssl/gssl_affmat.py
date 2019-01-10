import numpy as np
import scipy.spatial.distance as scipydist
from sklearn.neighbors import NearestNeighbors
from functools import partial

class AffMatGenerator(object):
    
    
    def generateAffMat(self,X):
        print("Creating mask...")
        K = self.mask_func(X)
        print("Done!")
        print("Calculating distances...")
        W =  np.reshape([0 if x == 0 else self.dist_func(x) for x in  np.reshape(K,(-1))],K.shape)
        print("Done!")
        assert(W.shape == (K.shape))
        print(W)
        return(W)

    
    def __init__(self, dist_func= "gaussian", mask_func="eps", metric="euclidean",**arg):
            self.metric = metric
            
            if dist_func == "gaussian":
                if not "sigma" in arg:
                    raise ValueError("Did not specify sigma for gaussian")
                
                den = 2*arg["sigma"]*arg["sigma"]
                self.dist_func = partial(lambda d,den: np.exp(-(d*d)/den),den=den)
            elif dist_func == "constant":
                self.dist_func = lambda d: 1
            else:
                self.dist_func = lambda d:d
            
            if mask_func == "eps":
                if not "sigma" in arg:
                    raise ValueError("Did not specify eps parameter for epsilon-neighborhood")
                self.mask_func = partial(lambda X,eps: epsilonMask(X, eps, self.metric),eps=arg["eps"])
            elif mask_func == "knn":
                if not "sigma" in arg:
                    raise ValueError("Did not specify k parameter for knn-neighborhood")
                self.mask_func = partial(lambda X,k: knnMask(X, k, self.metric),k=arg["k"])


def epsilonMask(X,eps,metric="euclidean"):
    print(type(X))
    assert isinstance(X, np.ndarray)
    
    K = scipydist.cdist(X,X)
    rows,cols = np.where(K > eps)
    K[rows,cols] = 0
    return(K)
    
def knnMask(X,k,symm = True):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    K = np.zeros((X.shape[0],X.shape[0]))
    for i in np.arange(X.shape[0]):
        distances, indices = nbrs.kneighbors([X[i,]])
        for dist, index in zip(distances,indices):
            K[i,index] = np.array(dist)
            if symm:
                K[index,i] = np.array(dist)
    return K
    
