import argparse
import os
import tensorflow as tf
from model import GraphSSL
import numpy as np
from multiprocessing import Process,Manager
import psutil
import sys
import progressbar
import pandas as pd
from plotly.io import orca as orca

FLAGS = tf.flags.FLAGS
'''
#Extras
tf.flags.DEFINE_string("dataset","spiral_hard","chosen dataset")
tf.flags.DEFINE_string("algorithm","LGC","which algorithm to use {LGC,GTAM,LDST,RF}")
tf.flags.DEFINE_float("split_test",10/1500,"Percent of dataset to be used for testing")
tf.flags.DEFINE_bool("can_plot",True,"enable plotting ops")
#Affmat
tf.flags.DEFINE_integer("aff_k",15,"(Affinity Matrix) number of nearest neighbors to use")
tf.flags.DEFINE_float("aff_sigma",0.2,"(Affinity Matrix) [aff_dist_func == gaussian] sensitivity of gaussian")
tf.flags.DEFINE_string("aff_dist_func","gaussian","(Affinity Matrix) which distance function to use")
#LGC
tf.flags.DEFINE_float("alpha",0.2,"(LGC algo.) hyperparameter alpha for LGC, with value inside [0,1].")
#GTAM/LDST
tf.flags.DEFINE_float("mu",99.0,"(GTAM/LDST algo.) hyperparameter alpha for GTAM/LDST that correlates with importance of fitting labels objective")
tf.flags.DEFINE_float("tuning_iter",-1,"(LDST algo.) Number of tuning iterations. If {-1, then use the")
'''
NUM_PROCESS=8
OUTPUT_NAME = 'results'
DISABLE_MATRIX_DICT = True

DEBUG_MODE = False


def __progress(q,total):
    
    print("TOTAL:{}".format(total))
    
    bar = progressbar.ProgressBar(max_value=total)
    counter = 0
    bar.update(0)
    
    while counter < total-1:
        q.get()
        counter += 1
        bar.update(counter)

def __run_exp(d, cfgs,prog_list, pid, cfgs_keys,q,WRITE_FREQ=10, overwrite=True):
    
    df = pd.DataFrame(index=range(WRITE_FREQ), columns=list(cfgs_keys).extend(["acc","elapsed_time"]))
    
     
    def check_mem():
        # gives an object with many fields
        vm = psutil.virtual_memory()
        return(vm.percent)
    def pack_affmat_str(i):
        #Extract info related to affmat
        affmat_keys = []
        affmat_vals = []
        for k,v in cfgs[i].items():
            if k.startswith("aff_") or k == "dataset":
                affmat_keys.append(k)
                affmat_vals.append(v)
        affmat_keys, affmat_vals = (list(t) for t in zip(*sorted(zip(affmat_keys, affmat_vals))))
        affmat_str = ""
        #Create string identiiiifying this affmat config
        for i in range(len(affmat_keys)):
            affmat_str = affmat_str + affmat_keys[i] + "=" + str(affmat_vals[i]) + ";"
        return(affmat_str)
        
    print("Process Number:{}".format(pid))
    count = 0
    for i in range(len(cfgs)):
        
        
        # disable output
        nullwrite = open(os.devnull, 'w')   
        oldstdout = sys.stdout
        if not DEBUG_MODE:
            sys.stdout = nullwrite 
        
        #Get affmat string to index into affmat dict
        affmat_str = pack_affmat_str(i)
        
        #Update dataframe
        for k,v in cfgs[i].items():
            df.loc[i%WRITE_FREQ,k] = v 
        #Create GSSL object
        gssl_obj = GraphSSL(cfgs[i])
        
        #Get Affinity Matrix
        if DISABLE_MATRIX_DICT:
            W = gssl_obj.getAffMatrix()
        else:
            W = d.get(affmat_str,None)
            if W is None:
                W = gssl_obj.getAffMatrix()
                d[affmat_str] = W
                count += 1
        
        #Run algorithm
        res = gssl_obj.train(W)
        if orca.status.state == "running":
            orca.shutdown_server()
        
        del gssl_obj
        df.loc[i%WRITE_FREQ,"acc"] = res["acc"]
        df.loc[i%WRITE_FREQ,"elapsed_time"] = res["elapsed_time"]

        #Update results dict
        if i % WRITE_FREQ == (WRITE_FREQ-1) or i == (len(cfgs)-1):
            result_path = './results/' + OUTPUT_NAME + '_pid=' + str(pid)+'.csv'
            f_exists = os.path.isfile(result_path)
            f_mode = "a"
            if (f_exists and overwrite and i == WRITE_FREQ-1) or (not f_exists and i == WRITE_FREQ-1):
                f_mode = "w"
            with open(result_path, f_mode) as f:            
                is_header = (f_mode == "w")
                df.iloc[0:((i % WRITE_FREQ)+1),:].to_csv(f, header= is_header)
                df.iloc[0:((i % WRITE_FREQ)+1),:] = np.nan
        
        #Update progress list
        prog_list[pid] = i/len(cfgs)
        
        sys.stdout = oldstdout
        
        #Enable output
        q.put(1)
        #if i % NUM_PROCESS == pid:
            #print(["%04f"% e for e in prog_list])
            #bar.update(np.mean(prog_list))
def runSetOfExperiments(config_dict_list = None):
    

    cfgs = [[0]*(1 + len(config_dict_list)//NUM_PROCESS)] * NUM_PROCESS
    cfgs = [[dict() for x in y] for y in cfgs]
    print(np.array(cfgs).shape) 

    cfgs_keys = set([])

    for i,cfg in enumerate(config_dict_list):
        assert len(cfgs[i % NUM_PROCESS][i //NUM_PROCESS].keys())==0
        cfgs[i % NUM_PROCESS][i //NUM_PROCESS] = dict(cfg)
        for k in config_dict_list[i].keys():
            cfgs_keys.add(k) 
            
    cfgs_keys = sorted(list(cfgs_keys))
    print("Keys found:{}".format(cfgs_keys))

    
    cfgs_size = [0] * NUM_PROCESS
    for i in range(NUM_PROCESS):
        cfgs[i] = list(filter(lambda x: len(x) > 0,cfgs[i]))
        cfgs_size[i] = len(cfgs[i])
    print(cfgs_size)
    assert sum(cfgs_size) == len(config_dict_list)
    
    manager = Manager()
    Affmat_dict = manager.dict()
    progress_list = manager.list([0]*NUM_PROCESS)
    q = manager.Queue()
    
    processes =  [Process(target=__run_exp, args = (Affmat_dict,cfgs[i],progress_list,i,cfgs_keys,q)) for i in range(NUM_PROCESS)]
    processes += [Process(target= __progress, args = (q,sum(cfgs_size)))]
    _ = [p.start() for p in processes]
    _ = [p.join() for p in processes]
    
if __name__ == '__main__':
    tf.app.run()