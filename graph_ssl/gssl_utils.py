import numpy as np
import sklearn.model_selection as skmm
import sklearn.manifold as skmf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def deg_matrix(W):
    return(np.diag(np.sum(W,axis=0)))

def init_matrix(Y,labeledIndexes):
    Y = Y - np.min(Y)
    M = np.max(Y)   
    def one_hot(x):
        oh = np.zeros((M+1))
        oh[x] = 1
        return oh 
    Y_0 = np.zeros((Y.shape[0],M+1))
    Y_0[labeledIndexes,:] = [one_hot(x) for x in Y[labeledIndexes]]
    return(Y_0)

def get_argmax(Y):
    return(np.argmax(Y,axis=1))


def split_indices(Y,split_p = 0.5):
    index_train, index_test  = skmm.train_test_split(np.arange(Y.shape[0]),
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
