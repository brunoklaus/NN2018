import numpy as np
import sklearn.model_selection as skmm
import sklearn.manifold as skmf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from plotly.api.v2.users import current

def deg_matrix(W):
    return(np.diag(np.sum(W,axis=0)))

def lap_matrix(W,is_normalized):

    if is_normalized:
            d_sqrt = np.reciprocal(np.sqrt(np.sum(W,axis=0)))
            d_sqrt[np.logical_not(np.isfinite(d_sqrt))] = 1
            d_sqrt = np.diag(d_sqrt)
            I = np.identity(W.shape[0])
            S = np.matmul(d_sqrt,np.matmul(W,d_sqrt))
            I = np.identity(W.shape[0])
            return( I - S )
    else:
        return( deg_matrix(W - W) )


def uniform_noise_mat(Y,p, deterministic=True):
    c = Y.shape[1]
    class_freq = [int(sum(Y[:,i])) for i in range(c)]
    
    
    A = np.zeros(shape=(c,c))

    for i in range(c):
        for j in range(c):
            A[i,j] = (1-p) if i == j else p/(c-1)
    #Fix possible isues
    if deterministic:
        for i in range(c):
            A[i,:] = np.round(A[i,:] * class_freq[i])
        
        A = A.astype(np.int32)
        print(A)
        for i in range(c):
            S = sum(A[i,:]) - class_freq[i]
            if S > 0:
                for s in range(S):
                    A[i,np.argmax(A[:,i])] -= 1
            elif S < 0: 
                print(S)
                for s in range(-S):
                    A[i,np.argmax(-A[:,i])] += 1
    return A
def apply_uniform_noise_deterministic(Y,labeledIndexes,p):
    old_Y = np.copy(Y)
    is_flat = np.ndim(Y) == 1
    if is_flat:
        Y = init_matrix(Y,labeledIndexes)
    c = Y.shape[1]
    n = Y.shape[0]
    
    Y = Y[labeledIndexes,:]
    Y_flat = np.argmax(Y,axis=1)

    vec = np.random.permutation(Y.shape[0])
    print(Y.shape)
    print(vec)
    assert not vec is None
    A = uniform_noise_mat(Y,p, deterministic=True)
    cursor = np.zeros((c),dtype=np.int32)
    print(A)

    for i in np.arange(Y_flat.shape[0]):
        current_class = Y_flat[vec[i]]
        while A[current_class,cursor[current_class]] == 0:
            cursor[current_class] += 1
            assert cursor[current_class] < c
        Y_flat[vec[i]] = cursor[current_class]
        A[current_class,cursor[current_class]] -= 1
 
    assert np.sum(A) == 0
    
    noisy_Y = np.zeros(shape=(n,c))
    labeledIndexes_where = np.where(labeledIndexes)[0]
    for l in range(Y_flat.shape[0]):
        noisy_Y[labeledIndexes_where[l],Y_flat[l]] = 1    
    print("Changed {} percent of entries".format(1-accuracy(np.argmax(Y,axis=1),Y_flat)))
    
    if is_flat:
        old_Y[labeledIndexes] = np.argmax(noisy_Y[labeledIndexes],axis=1)
        return old_Y
    else:
        return noisy_Y

def init_matrix(Y,labeledIndexes):
    Y = np.copy(Y)
    Y = np.array(Y) - np.min(Y)
    M = np.max(Y)   
    def one_hot(x):
        oh = np.zeros((M+1))
        oh[x] = 1
        return oh 
    Y_0 = np.zeros((Y.shape[0],M+1))
    Y_0[labeledIndexes,:] = [one_hot(x) for x in Y[labeledIndexes]]
    return(Y_0)

def init_matrix_argmax(Y):
    return(np.argmax(Y,axis=1))

'''
    Returns a percentage p of indices, using stratification
    @param Y the vector from which to split with stratification
    @param split_p the percentage of stratified indexes to return
    @return split_p percent of stratified indexes
'''
def split_indices(Y,split_p = 0.5):
    index_train, _  = skmm.train_test_split(np.arange(Y.shape[0]),
                                                     stratify=Y,test_size=1-split_p)
    b = np.zeros((Y.shape[0]),dtype=np.bool)
    b[index_train] = True
    return b

def accuracy_unlabeled(Y_pred,Y_true, labeled_indexes):
    unlabeled_indexes = np.logical_not(labeled_indexes)
    return(np.sum( (Y_pred[unlabeled_indexes] == Y_true[unlabeled_indexes]).\
                   astype(np.int32) )/Y_true[unlabeled_indexes].shape[0])
def accuracy(Y_pred,Y_true):
    return(np.sum( (Y_pred== Y_true).astype(np.int32) )/Y_true.shape[0])
    
def get_Isomap(X,n_neighbors = 5):
    return(skmf.Isomap(n_neighbors).fit_transform(X))

def get_PCA(X):
    return(PCA.fit_transform(X))

def get_Standardized(X):
    return(StandardScaler().fit_transform(X))


def calc_Z(Y, labeledIndexes,D,estimatedFreq=None,reciprocal=False):
            if estimatedFreq is None:
                estimatedFreq = np.repeat(1,Y.shape[0])
            if Y.ndim == 1:
                Y = init_matrix(Y,labeledIndexes)
            
            Z = np.array(Y)
            for i in np.where(labeledIndexes == True)[0]:
                if reciprocal:
                    Z[i,np.argmax(Y[i,:])] = 1/D[i]
                else:
                    Z[i,np.argmax(Y[i,:])] = D[i]

            for i in np.arange(Y.shape[1]):
                Z[:,i] = (Z[:,i] / np.sum(Z[:,i])) * estimatedFreq[i]
            return(Z)
