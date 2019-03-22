import os
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import sklearn.preprocessing as pp
import parallel_processing as paral

def prod(a,b):
    DICTS = []
    newvals = b[1]
    newkeys = b[0]
    for d in a:
        for i in np.arange(len(newvals)): 
            K = list(d.keys()) + [newkeys]
            V = list(d.values()) + [newvals[i]] 
            newDict = {}
            for k,v in zip(K,V):
                newDict[k] = v
            DICTS += [newDict]
    return(DICTS)

def allPermutations(args):
    return(reduce(lambda a,b: prod(a,b), list(zip(args.keys(),args.values())), [{}]))

#COmbine Two LISTS of dicts
def comb(dict_A,dict_B):
    if not isinstance(dict_A,list):
        raise "ERROR: dict_A, dict_B must each be a list of dicts"
    if not isinstance(dict_B,list):
        raise "ERROR: dict_A, dict_B must each be a list of dicts"
    
        
    l = [None] * (len(dict_A) * len(dict_B))
    i = 0
    for a in dict_A:
        for b in dict_B:
            comb_dict = {}
            for k,v in zip(list(a.keys()) + list(b.keys()), list(a.values()) + list(b.values())):
                comb_dict[k] = v
            l[i] = comb_dict
            i += 1
    assert i == (len(dict_A) * len(dict_B))
    return l

def run_program(args):
    #Run program with set learning rate
    comm1  = "python3 runSetOfExperiments.py " 
    for key in args.keys():
        comm1 = comm1 + " --" + key
        if args[key] != None:
            comm1 = comm1 + "=" + str(args[key])
            
    print(comm1)
    os.system(comm1 )
def run_all(args):
    num_proc = int(input('Total number of processes:'))
    p_id = int(input('Process ID (0 to num_proc - 1):'))
    args["run_id"] = np.arange(p_id, 10, num_proc)
    i = 0
    cmb = allPermutations(args)
    for x in cmb:
        print("({}/{})".format(i,len(cmb)))
        run_program(x)
        i += 1
        

extra_optionselect = allPermutations({
    "dataset": ["gaussian"],
    "dataset_sd": [0.4,1,2,3],
    "labeled_percent": [0.1],
    "can_plot": [False],
    "id": np.arange(20),
    "corruption_level" : [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    })

Affmat_optionselect_simple = allPermutations({
    "aff_k": [15],
    "aff_mask_func": ["knn"],
    "aff_sigma": [2],
    "aff_dist_func":["gaussian"]
})

Affmat_optionselect_0 = allPermutations({
    "aff_k": [5,7,15,100],
    "aff_mask_func": ["knn"],
    "aff_sigma": [0.01,0.2,0.5,6],
    "aff_dist_func":["gaussian"]
})

Affmat_optionselect_1 = allPermutations({
    "aff_k": [5,7,15,100],
    "aff_mask_func": ["knn"],
    "aff_dist_func":["LNP"]
})

Affmat_optionselect_2 = allPermutations({
    "aff_k": [3,5,10,15,20],
    "aff_dist_func":["euclidean","constant"]
})


Alg_optionselect_LGC = allPermutations({
   "algorithm" : ["LGC"],
   "alpha" : [0.0001,0.2,0.9,0.999]
})
Alg_optionselect_GTAM = allPermutations({
   "algorithm" : ["GTAM"],
   "tuning_iter" : [0],
   "mu":[99.0]  
})

Alg_optionselect_RF = allPermutations({
   "algorithm" : ["RF"],
})
Alg_optionselect_LP = allPermutations({
   "algorithm" : ["LP"],
})


Alg_optionselect_LDST = allPermutations({
   "algorithm" : ["LDST"],
   "tuning_iter" : [0.25,0.5,1,2],
   "mu":[99.0]  
})

def test_RF():    
    extra_optionselect_RF = allPermutations({
        "dataset": ["gaussian"],
        "dataset_sd" : [0.4,1,2,3],
        "labeled_percent": [0.01,0.05,0.1,0.5],
        "can_plot": [False],
        "id": np.arange(20),
        "corruption_level" : [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
        })
    opt = comb(Affmat_optionselect_simple,Alg_optionselect_RF)
    opt = comb (extra_optionselect_RF,opt)
    print("opt size: {}".format(len(opt)))
    paral.OUTPUT_NAME = 'results_RF'
    paral.runSetOfExperiments(opt)
    ###################################3

def test():
    opt = comb(Affmat_optionselect_simple, Alg_optionselect_LP + Alg_optionselect_LGC + Alg_optionselect_LDST + Alg_optionselect_GTAM + Alg_optionselect_RF)
    opt = comb (extra_optionselect,opt)
    print("opt size: {}".format(len(opt)))
    paral.runSetOfExperiments(opt)

def test_LDST():    
    extra_optionselect_RF = allPermutations({
        "dataset": ["spiral","spiral_hard","gaussians_sd=1","gaussians_sd=2","gaussians_sd=3","gaussians_sd=0.4"],
        "labeled_percent": [0.01,0.05,0.1,0.5],
        "can_plot": [False],
        "id": np.arange(20),
        "corruption_level" : [0,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
        })
    opt = comb(Affmat_optionselect_simple,Alg_optionselect_LDST)
    opt = comb (extra_optionselect_RF,opt)
    print("opt size: {}".format(len(opt)))
    paral.OUTPUT_NAME = 'results_RF'
    paral.runSetOfExperiments(opt)
    ###################################3


def test_plot():
    Alg_optionselect_LDST = allPermutations({
   "algorithm" : ["LDST"],
   "tuning_iter" : [1],
   "mu":[99.0]  
})
    Alg_optionselect_LGC = allPermutations({
   "algorithm" : ["LGC"],
   "alpha" : [0.999]
})
    extra_optionselect_temp = allPermutations({
    "dataset": ["spiral","spiral_hard"],
    "labeled_percent": [0.1],
    "can_plot": [True],
    "id": np.arange(1),
    "corruption_level" : [0.35]
    })
    opt = comb(Affmat_optionselect_simple,Alg_optionselect_LDST)
    opt = comb (extra_optionselect_temp,opt)
    print("opt size: {}".format(len(opt)))
    paral.OUTPUT_NAME = 'results_RF'
    paral.NUM_PROCESS = 1
    paral.runSetOfExperiments(opt)
#########################################

test_plot()
