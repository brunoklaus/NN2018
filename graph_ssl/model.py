
import toy_datasets as toyds
import plot as plt
import numpy as np
import gssl_utils as gutils
import pandas as pd
import time
from gssl_affmat import AffMatGenerator
from sklearn.ensemble import RandomForestClassifier
from gssl_utils import calc_Z
from reportlab.lib.validators import isInstanceOf
from sklearn.datasets import make_blobs, make_moons
from math import sqrt

class GraphSSL(object):
    def __init__(self,  args):
        self.args = args
        self.can_plot = args["can_plot"]

        
        if self.args["dataset"] == "gaussian":
            ds_x,ds_y =  make_blobs(n_samples=1000, n_features=2,\
                                    centers=[[0,0],[sqrt(2),sqrt(2)]], cluster_std=self.args["dataset_sd"],\
                                     shuffle=True)
            self.dataset = {"X":ds_x,"Y":ds_y}
        elif self.args["dataset"] == "spiral" and "dataset_sd" in self.args:
            ds_x,ds_y =  make_moons(n_samples=1000,noise=self.args["dataset_sd"],\
                                     shuffle=True)
            self.dataset = {"X":ds_x,"Y":ds_y}
            
        else:
            self.dataset = toyds.getDataframe(args["dataset"])

        if self.dataset["X"].shape[1] > 3:
            self.transformed_dataset = gutils.get_Isomap(self.dataset["X"], 10)
        else:
            self.transformed_dataset = self.dataset
        self.transformed_dataset = gutils.get_Standardized(self.transformed_dataset["X"])
    
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
             constant_prop=False):
        
        
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
        assert total_iter + tuning_iter > tuning_iter
        for i in np.arange(total_iter + tuning_iter):
            is_tuning = (i < tuning_iter)
            
            
            if i == tuning_iter:
                Y_tuned = np.array(Y)
                labeledIndexes_tuned = np.array(labeledIndexes)
            
            '''Z matrix - The binary values of current Y are replaced with their corresponding D entries.
                Then, we normalize each row so that row sums to its estimated influence
            '''
            Z = gutils.calc_Z(Y, labeledIndexes, D, estimatedFreq,reciprocal=False)

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
            if is_tuning and constant_prop:
                id_min_line = np.argmin(Q[:,id_max_col])
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
            
    def LP(self,W,Y,labeledIndexes):
        Y = np.copy(Y)
        if Y.ndim == 1:
            Y[np.logical_not(labeledIndexes)] = 0
            Y = gutils.init_matrix(Y,labeledIndexes)
        if not W.shape[0] == Y.shape[0]:
            raise ValueError("W,Y shape not compatible")
        
        u = np.reshape(np.array(np.where(np.logical_not(labeledIndexes))),(-1))
        l = np.reshape(np.array(np.where(labeledIndexes)),(-1))
        
        d_inv = np.reciprocal(np.sum(W,axis=0))
        d_inv[np.logical_not(np.isfinite(d_inv))] = 1
        d_inv = np.diag(d_inv)
        
        P  = d_inv @ W
        
        I = np.identity(Y.shape[0] - sum(labeledIndexes))
        
        P_ul = P[u[:, None],l]
        P_uu = P[u[:, None],u]
        
        Y[u,:] = np.linalg.inv(I - P_uu) @ P_ul @ Y[l,:]
        return(Y)
    
    def build_model(self):
        return None
    
    
    def experiment_LDST(self,X_transformed,Y,W,labeledIndexes, mu = 99.0, tuning_iter=2, plot=True,Y_noisy=None):
        
        if Y_noisy is None:
            Y_noisy = Y
                
        classif_LDST, Y_tuned, l_tuned = self.LDST(W=W, Y=Y_noisy, mu=mu, labeledIndexes=labeledIndexes,tuning_iter=tuning_iter)
        classif_LDST = np.argmax(classif_LDST,axis=1)
        
        
        
        
        
        if self.args["can_plot"] and plot:
            ''' PLOT Tuned Y '''   
            vertex_opt = plt.vertexplotOpt(Y=Y_tuned,mode="discrete",size=7,labeledIndexes=l_tuned)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LDST: Tuned Y")
            ''' PLOT LDST result '''             
            vertex_opt = plt.vertexplotOpt(Y=classif_LDST,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LDST result")
        
        acc = gutils.accuracy(classif_LDST, Y)
        
        return acc
        
    def experiment_LGC(self,X_transformed,Y,W,labeledIndexes, alpha = 0.9, plot=True,Y_noisy=None ):  
        if Y_noisy is None:
            Y_noisy = Y      
        classif_LGC = np.argmax(self.LGC(W=W, Y=Y_noisy, labeledIndexes=labeledIndexes, alpha=alpha),axis=1)
        
        if self.args["can_plot"] and plot:                
            vertex_opt = plt.vertexplotOpt(Y=classif_LGC,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LGC result")
        
        acc = gutils.accuracy(classif_LGC, Y)
        return acc
        

    def experiment_LP(self,X_transformed,Y,W,labeledIndexes, plot=True,Y_noisy=None ):  
        if Y_noisy is None:
            Y_noisy = Y      
        classif_LP = np.argmax(self.LP(W=W, Y=Y_noisy, labeledIndexes=labeledIndexes),axis=1)
        
        if self.args["can_plot"] and plot:                
            vertex_opt = plt.vertexplotOpt(Y=classif_LP,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="LP result")
        
        acc = gutils.accuracy(classif_LP, Y)
        return acc
    
    def experiment_RF(self,X,X_transformed,Y,W,labeledIndexes, plot=True,Y_noisy=None):
        if Y_noisy is None:
            Y_noisy = Y

        #RF Classifier
        rf = RandomForestClassifier(n_estimators=100).fit(X[labeledIndexes,:],Y_noisy[labeledIndexes])
        rf_pred = np.array(rf.predict(X))
        if self.args["can_plot"] and plot:                
            vertex_opt = plt.vertexplotOpt(Y=rf_pred,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="RF result")
        acc = gutils.accuracy(rf_pred, Y)
        return acc    
        
        
    def plot_labeled_indexes(self,X_transformed,Y,W,labeledIndexes,title="labeled indexes"):  
        #Plot 1: labeled indexes
        if self.args["can_plot"]:       
            
            vertex_opt = plt.vertexplotOpt(Y=Y,mode="discrete",size=7,labeledIndexes=labeledIndexes)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title=title)
            
            
    
    
    def plot_true_classif(self,X_transformed,Y,W):
        #Plot 2: True classif
        if self.args["can_plot"]:          
            vertex_opt = plt.vertexplotOpt(Y=Y,mode="discrete",size=7)
            plt.plotGraph(X_transformed,plot_dim=2,W=W, vertex_opt= vertex_opt,edge_width=1,\
                          interactive = False,title="True classes")

    
    
    def getAffMatrix(self):
        X = gutils.get_Standardized(self.dataset["X"])
                
        affmat_dict = dict()
        for k,v in self.args.items():
            if k.startswith("aff_"):
                affmat_dict[k[4:]] = v
    
        W = AffMatGenerator(**affmat_dict).generateAffMat(X)
        return(W)
        
            
    def train(self,W=None):
        Y = self.dataset["Y"]
        labeledIndexes = gutils.split_indices(Y, self.args["labeled_percent"])
        
        Y_noisy = gutils.apply_uniform_noise_deterministic(self.dataset["Y"],labeledIndexes,self.args["corruption_level"])
        
        if "tuning_iter" in self.args.keys():
            self.args["tuning_iter"] = int(self.args["tuning_iter"] * self.args["corruption_level"] * self.args["labeled_percent"] * Y.shape[0])

        
        if W is None:
            W = self.getAffMatrix()
                
        
        self.plot_labeled_indexes(self.transformed_dataset, self.dataset["Y"], W, labeledIndexes)
        self.plot_labeled_indexes(self.transformed_dataset, Y_noisy, W, labeledIndexes,title="noisy Y")
        
        switcher = {
            "LGC": lambda :  self.experiment_LGC(self.transformed_dataset, Y, W, labeledIndexes, alpha = self.args["alpha"],\
                                                 plot=self.args["can_plot"],Y_noisy =Y_noisy),
            "GTAM":lambda: self.experiment_LDST(self.transformed_dataset,Y,W,labeledIndexes, mu = self.args["mu"], tuning_iter=0, \
                                                plot=self.args["can_plot"],Y_noisy=Y_noisy),
            "LDST":lambda: self.experiment_LDST(self.transformed_dataset,Y,W,labeledIndexes, mu = self.args["mu"], tuning_iter= self.args["tuning_iter"],\
                                                 plot=self.args["can_plot"],Y_noisy=Y_noisy),
            "RF": lambda :  self.experiment_RF(self.dataset["X"],self.transformed_dataset, Y, W, labeledIndexes,plot=self.args["can_plot"],Y_noisy=Y_noisy),\
            
            "LP": lambda: self.experiment_LP(self.transformed_dataset, Y, W, labeledIndexes, plot=self.args["can_plot"], Y_noisy=Y_noisy)
        }
        start = time.time()
        acc = switcher[self.args["algorithm"]]()
        end = time.time()
        print("Accuracy {}:{}".format(self.args["algorithm"],acc))
        return {"acc":acc,"elapsed_time": (end-start)} 
         
        

        
        

        
    