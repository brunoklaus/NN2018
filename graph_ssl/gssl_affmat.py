import numpy as np
import scipy.spatial.distance as scipydist
from sklearn.neighbors import NearestNeighbors
from functools import partial
import quadprog
import sys
import progressbar

class AffMatGenerator(object):
    
    
    def generateAffMat(self,X):
        print("Creating Affinity Matrix...")
        if self.dist_func_str == "LNP" or self.dist_func_str == "NLNP":
            W = self.mask_func(X)
        else:
            K = self.mask_func(X)
            W =  np.reshape([0 if x == 0 else self.dist_func(x) for x in  np.reshape(K,(-1))],K.shape)
        print("Done!")
        assert(W.shape == (X.shape[0],X.shape[0]))
        assert(not np.all(W==0))
        return(W)

    
    def __init__(self, dist_func= "gaussian", mask_func="eps", metric="euclidean",**arg):
            self.metric = metric
            self.dist_func_str = dist_func
            
            if dist_func in ["LNP","NLNP"]:
                mask_func = dist_func
            if mask_func in ["LNP","NLNP"]:
                dist_func = mask_func
            
            
            
            if dist_func == "gaussian":
                if not "sigma" in arg:
                    raise ValueError("Did not specify sigma for gaussian")
            
                den = 2*arg["sigma"]*arg["sigma"]
                self.dist_func = partial(lambda d,den: np.exp(-(d*d)/den),den=den)
            elif dist_func == "constant":
                self.dist_func = lambda d: 1
            else:
                self.dist_func = lambda d: 1/d
            
            if mask_func == "eps":
                if not "sigma" in arg:
                    raise ValueError("Did not specify eps parameter for epsilon-neighborhood")
                self.mask_func = partial(lambda X,eps: epsilonMask(X, eps),eps=arg["eps"])
            elif mask_func == "knn":
                if not "sigma" in arg:
                    raise ValueError("Did not specify k parameter for knn-neighborhood")
                self.mask_func = partial(lambda X,k: knnMask(X, k),k=arg["k"])
            elif mask_func == "LNP":
                if not "k" in arg:
                    raise ValueError("Did not specify k for LNP")
                self.mask_func = partial(lambda X,k: LNP(X, k),k=arg["k"])
            elif mask_func == "NLNP":
                if not "k" in arg:
                    raise ValueError("Did not specify k for NLNP")
                self.mask_func = partial(lambda X,k: NLNP(X, k),k=arg["k"])

'''
    Calculates the binary matrix K corresponding to a epsilon-neighborhood
    @param X A matrix of shape (n,d) representing n vectors of d dimensions
    @param eps A parameter such that K[i,j] = 1 iff dist(X_i,X_j) < eps
    @return K a binary matrix of (n,d) corresponding to the eps-neighborhood graph
'''
def epsilonMask(X,eps):
    print(type(X))
    assert isinstance(X, np.ndarray)
    
    K = scipydist.cdist(X,X)
    rows,cols = np.where(K > eps)
    K[rows,cols] = 0
    return(K)
'''
    Calculates the binary matrix K corresponding to a knn-neighborhood
    @param X A matrix of shape (n,d) representing n vectors of d dimensions
    @param k A parameter such that K[i,j] = 1 iff X_i is one of the k-nearest neighbors of X_j
    @param symm if True, then K[i,j] = max(K[i,j],K[j,i])
    @return K a binary matrix of (n,d) corresponding to the knn graph 
'''   
def knnMask(X,k,symm = True):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    K = np.zeros((X.shape[0],X.shape[0]))
    for i in np.arange(X.shape[0]):
        distances, indices = nbrs.kneighbors([X[i,]])
        
        for dist, index in zip(distances,indices):
            K[i,index] = np.array(dist)
            if symm:
                K[index,i] = np.array(dist)
    return K
    

def __quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]   

def LNP(X,k, symm = True):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    W = np.zeros((X.shape[0],X.shape[0]))
    
    q = np.zeros((k))
    
    G = -np.identity(k)
    h = np.zeros((k))
    
    A = np.ones((1,k))
    b = np.ones((1))
    
    
    for i in np.arange(X.shape[0]):
        _, indices = nbrs.kneighbors([X[i,]])
        indices = indices[0][1:]
        P = np.zeros([k,k])
        for m in range(k):
            for n in range(k):
                P[m,n] = np.dot((X[i,]-X[indices[m],]),(X[i,]-X[indices[n],]).T)        
        for m in range(k):
            P[m,m] += 1e-03
          
        W_lnp = __quadprog_solve_qp(P, q, G, h, A, b)
        
        for m in range(k):
            W[i,indices[m]] = W_lnp[m]
    if symm:
        W = 0.5*(W + W.T)
    return(W)

def NLNP(X,k, symm = True):
    W = LNP(X,k,symm)
    for i in range(W.shape[0]):
        nonz = (np.where(W[i,] > 0))
        W[i,nonz] = np.reciprocal(W[i,nonz])
        W[i,nonz] = W[i,nonz] / np.linalg.norm(W[i,nonz])
    return(W)


 
def knnMask_sparse(X,k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    K = np.zeros((X.shape[0],k,2),dtype=np.int32)
    Ds = np.zeros((X.shape[0],k))
    print(K.shape)
    for i in progressbar.progressbar(np.arange(X.shape[0])):
        dist, ind = nbrs.kneighbors([X[i,]])
        dist, ind = dist[0], ind[0]
        for j in range(k):
            K[i,j,:] = [i, ind[j]]
            Ds[i,j] = dist[j]
    return K, Ds

def teste(NUM_N,NUM_D,NUM_K):
    print("Generating....")
    X = np.random.normal(size=(NUM_N,NUM_D))
    nbrs = NearestNeighbors(n_neighbors=NUM_K+1, algorithm='brute').fit(X)
    #K = np.zeros((X.shape[0],X.shape[0]))
    for i in progressbar.progressbar(np.arange(X.shape[0])):
        distances, indices = nbrs.kneighbors([X[i,]])
        
#teste(50000,256,100)
