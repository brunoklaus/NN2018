
import toy_datasets as toyds
import plot as plt
import numpy as np
import graph_ssl.gssl_utils as gutils
import pandas as pd

from graph_ssl.gssl_affmat import AffMatGenerator
from sklearn.ensemble import RandomForestClassifier
from cProfile import label

class GraphSSL(object):
    
    
    def LGC(self,W,Y,labeledIndexes, alpha = 0.1):
        if Y.ndim == 1:
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
    
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.can_plot = args.can_plot
        self.split_test = args.split_test
        
        self.dataset = toyds.getDataframe(args.dataset)
        
    
    def build_model(self):
        return None
    
    def train(self):
        Y = self.dataset["Y"]
        X = gutils.get_Standardized(self.dataset["X"])
        W = AffMatGenerator(sigma=2,k=3,mask_func="knn",dist_func="gaussian").generateAffMat(X)
        
        labeledIndexes = gutils.split_indices(Y, self.split_test)
        
        if X.shape[1] > 3:
            X_transformed  =  gutils.get_Isomap(X, 3)
            print(X_transformed.shape)
        else:
            X_transformed = X
        #Plot 1: labeled indexes
        if self.can_plot:          
            vertex_opt = plt.vertexplotOpt(Y=Y,mode="discrete",size=7,labeledIndexes=labeledIndexes)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="labeled indexes")
        #Plot 2: True classif
        if self.can_plot:          
            vertex_opt = plt.vertexplotOpt(Y=Y,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="True classes")
            
        
         
       
        #Plot 3: LGC result
        classif_LGC = gutils.get_argmax(self.LGC(W=W, Y=Y, labeledIndexes=labeledIndexes, alpha=0.2))
        if self.can_plot:                
            vertex_opt = plt.vertexplotOpt(Y=classif_LGC,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LGC result")
        
        print("Accuracy LGC:{}".format(gutils.accuracy_unlabeled(classif_LGC, Y,labeledIndexes)))
        
        #RF Classifier
        rf = RandomForestClassifier().fit(X[labeledIndexes,:],Y[labeledIndexes])
        rf_pred = np.array(rf.predict(X))
        print("Accuracy RF:{}".format(gutils.accuracy_unlabeled(rf_pred, Y,labeledIndexes)))
        if self.can_plot:                
            vertex_opt = plt.vertexplotOpt(Y=rf_pred,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="RF result")
        
    