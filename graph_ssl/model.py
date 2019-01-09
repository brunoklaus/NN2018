
import toy_datasets as toyds
import plot as plt
import numpy as np
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
                                        mode="constant",size=6)
        
        W = np.random.random_integers(0,10000,np.square(num_samples))
        W[np.where(W < 10000)] = 0
        W = np.reshape(W,(num_samples,num_samples))

        plt.plotGraph(self.dataset["X"],W=W,plot_dim=2, vertex_opt= vertex_opt)
        