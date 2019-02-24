import numpy as np
import sklearn.model_selection as skmm
import sklearn.manifold as skmf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def calc_Z(Y, labeledIndexes,D,estimatedFreq=None):
            if estimatedFreq is None:
                estimatedFreq = np.repeat(1,Y.shape[0])
            if Y.ndim == 1:
                Y = init_matrix(Y,labeledIndexes)
            
            Z = np.array(Y)
            for i in np.where(labeledIndexes == True)[0]:
                
                Z[i,np.argmax(Y[i,:])] = D[i]
                
            for i in np.arange(Y.shape[1]):
                Z[:,i] = (Z[:,i] / np.sum(Z[:,i])) * estimatedFreq[i]
            return(Z)
