
import toy_datasets as toyds
import plot as plt
import numpy as np
import graph_ssl.gssl_utils as gutils
import pandas as pd

from graph_ssl.gssl_affmat import AffMatGenerator
from sklearn.ensemble import RandomForestClassifier
from cProfile import label
from graph_ssl.gssl_utils import calc_Z

class GraphSSL(object):
    
    
    def LGC(self,W,Y,labeledIndexes, alpha = 0.1):
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y[np.logical_not(labeledIndexes)] = 0
            Y = gutils.init_matrix(Y,labeledIndexes)
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        #Get D^{-1/2}       
        d_sqrt = np.reciprocal(np.sqrt(np.sum(W,axis=0)))
        d_sqrt[np.logical_not(np.isfinite(d_sqrt))] = 1
        d_sqrt = np.diag(d_sqrt)
        
        
        S = np.matmul(d_sqrt,np.matmul(W,d_sqrt))
        I = np.identity(Y.shape[0])
        return(np.matmul(np.linalg.inv(I - alpha*S),Y))
    
    def LDST(self,W,Y,labeledIndexes,mu = 99.0,useEstimatedFreq=True,total_iter = None,tuning_iter = 0,
             flip_same_class=False):
        Y_old = np.copy(Y)
        labeledIndexes_old = np.copy(labeledIndexes)
        
        Y = np.copy(Y)
        #We make a copy of labeledindexes
        labeledIndexes = np.array(labeledIndexes)
        
        if Y.ndim == 1:
            Y = gutils.init_matrix(Y,labeledIndexes)
           
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        num_labeled = Y[labeledIndexes].shape[0]
        num_unlabeled = Y.shape[0] - num_labeled
        num_classes = Y.shape[1]
        
        D = np.sum(W,axis=0)
        if useEstimatedFreq:
            estimatedFreq = np.sum(Y[labeledIndexes],axis=0) / num_labeled
        else:
            estimatedFreq = np.repeat(1/num_classes,num_classes)
        
        #Identity matrix
        I = np.identity(W.shape[0])
        #Get graph laplacian
        L = gutils.lap_matrix(W, is_normalized=True)
        #Propagation matrix
        P = np.linalg.inv( I + 0.5*(L + L.transpose())/mu )
        P_t = P.transpose()
        #Matrix A
        A = ((P_t @ L) @ P) + ((P_t - I) @ (P - I))
        A = A + A.transpose()
        
        Z = []
        
        if total_iter is None:
            #Do the full thing
            total_iter = num_unlabeled
        else:
            total_iter = min(total_iter,num_unlabeled)
        
        for i in np.arange(total_iter + tuning_iter):
            is_tuning = (i < tuning_iter)
            if is_tuning:
                continue
            
            if i == tuning_iter:
                Y_tuned = np.array(Y)
                labeledIndexes_tuned = np.array(labeledIndexes)
            
            '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                Then, we normalize each row so that row sums to its estimated influence
            '''
            Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq)
            #Compute graph gradient
            Q = np.matmul(A,Z)
            
            #During label tuning, we'll also 'unlabel' the argmax
            if is_tuning:
                unlabeledIndexes = np.logical_not(labeledIndexes)
                temp = Q[unlabeledIndexes,:]
                Q[unlabeledIndexes,:] = -np.inf
                id_max = np.argmax(Q)
                id_max_line = id_max // num_classes
                id_max_col = id_max % num_classes
            
            
            
            if is_tuning:
                Q[unlabeledIndexes,:] = temp
                
            Q[labeledIndexes,:] = np.inf
            #Find minimum unlabeled index
            if is_tuning and flip_same_class:
                id_min_line = np.argmin(np.reshape(Q[:,id_max_col], (Q.shape[0],1)))
                id_min_col = id_max_col
            else:
                id_min = np.argmin(Q)
                id_min_line = id_min // num_classes
                id_min_col = id_min % num_classes
            
            
 
            
            
            #Update Y and labeledIndexes
            labeledIndexes[id_min_line] = True
            Y[id_min_line,id_min_col] = 1
            if is_tuning:
                labeledIndexes[id_max_line] = False
                Y[id_max_line,id_max_col] = 0
            
        if tuning_iter > 0:
            return P@Z, np.argmax(Y_tuned,axis=1), labeledIndexes_tuned 
        else:
            return P@Z, Y_old, labeledIndexes_old
            
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.can_plot = args.can_plot
        self.split_test = args.split_test
        
        self.dataset = toyds.getDataframe(args.dataset)
        
    
    def build_model(self):
        return None
    
    
    def experiment_LDST(self,X_transformed,Y,W,labeledIndexes, mu = 99.0, tuning_iter=2, plot=True):        
        classif_LDST, Y_tuned, l_tuned = self.LDST(W=W, Y=Y, mu=mu, labeledIndexes=labeledIndexes,tuning_iter=tuning_iter)
        classif_LDST = gutils.get_argmax(classif_LDST)
        
        if self.can_plot and plot:
            ''' PLOT Tuned Y '''   
            vertex_opt = plt.vertexplotOpt(Y=Y_tuned,mode="discrete",size=7,labeledIndexes=l_tuned)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LDST: Tuned Y")
            ''' PLOT LDST result '''             
            vertex_opt = plt.vertexplotOpt(Y=classif_LDST,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LDST result")
        
        acc = gutils.accuracy_unlabeled(classif_LDST, Y,labeledIndexes)
        
        return acc
        
    
    def experiment_LGC(self,X_transformed,Y,W,labeledIndexes, alpha = 0.9, plot=True):        
        classif_LGC = gutils.get_argmax(self.LGC(W=W, Y=Y, labeledIndexes=labeledIndexes, alpha=alpha))
        
        if self.can_plot and plot:                
            vertex_opt = plt.vertexplotOpt(Y=classif_LGC,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LGC result")
        
        acc = gutils.accuracy_unlabeled(classif_LGC, Y,labeledIndexes)
        return acc
        
    def plot_labeled_indexes(self,X_transformed,Y,W,labeledIndexes):  
        #Plot 1: labeled indexes
        if self.can_plot:          
            vertex_opt = plt.vertexplotOpt(Y=Y,mode="discrete",size=7,labeledIndexes=labeledIndexes)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="labeled indexes")
            
    def plot_true_classif(self,X_transformed,Y,W):
        #Plot 2: True classif
        if self.can_plot:          
            vertex_opt = plt.vertexplotOpt(Y=Y,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="True classes")

    def experiment_RF(self,X,X_transformed,Y,W,labeledIndexes):
        #RF Classifier
        rf = RandomForestClassifier(n_estimators=100).fit(X[labeledIndexes,:],Y[labeledIndexes])
        rf_pred = np.array(rf.predict(X))
        if self.can_plot:                
            vertex_opt = plt.vertexplotOpt(Y=rf_pred,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="RF result")
        acc = gutils.accuracy_unlabeled(rf_pred, Y,labeledIndexes)
        return acc
            
    def train(self):
        Y = self.dataset["Y"]
        X = gutils.get_Standardized(self.dataset["X"])
        W = AffMatGenerator(sigma=0.1,k=10,mask_func="knn",dist_func="gaussian").generateAffMat(X)
        
        labeledIndexes = gutils.split_indices(Y, self.split_test)
        
        if X.shape[1] > 3:
            X_transformed  =  gutils.get_Isomap(X, 5)
        else:
            X_transformed = X
        
        Y_old = np.copy(Y)
                
        acc_LDST = self.experiment_LDST(X_transformed, Y, W, labeledIndexes, mu = 99.0, tuning_iter = 0)
        print("Accuracy LDST:{}".format(acc_LDST))
        
        acc_LGC = self.experiment_LGC(X_transformed, Y, W, labeledIndexes, alpha = 0.8)
        print("Accuracy LGC:{}".format(acc_LGC))

        acc_RF = self.experiment_RF(X, X_transformed, Y, W, labeledIndexes)
        print("Accuracy RF:{}".format(acc_RF))
            
        assert(np.all(Y_old == Y))
        
         
        

        
        

        
    