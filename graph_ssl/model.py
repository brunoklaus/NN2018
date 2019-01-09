
import toy_datasets as toyds
import plot as plt
import numpy as np

from graph_ssl.gssl_affmat import AffMatGenerator
class GraphSSL(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        
        self.dataset = toyds.getDataframe(args.dataset)
        
    
    def build_model(self):
        return None
    
    def train(self):
        num_samples = self.dataset["Y"].shape[0]
        vertex_opt = plt.vertexplotOpt(Y=self.dataset["Y"],
                                        mode="constant",size=2)
        
        
        W = AffMatGenerator(sigma=1,k=10,mask_func="knn").generateAffMat(self.dataset["X"])

        

        plt.plotGraph(self.dataset["X"],W=W,plot_dim=3, vertex_opt= vertex_opt,edge_width=1,\
                      interactive = True)
        