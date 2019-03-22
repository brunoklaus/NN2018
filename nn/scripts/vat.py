
import os
import tensorflow as tf
from dataset_utils import convert_images_and_labels_and_emb
from gssl_affmat import AffMatGenerator
import sup_classifier
import layers as L
import numpy as np
from datetime import datetime
import gssl_utils as gutils
import time
from progressbar import progressbar
from python_utils.time import epoch
from tensorflow.contrib.tensorboard.plugins import projector
import toy_datasets as toy
from create_sparse_mat import load_whole_dataset, load_whole_affmat
import os
import plot as sslplot
from sphinx.ext.todo import Todo
from docutils.nodes import Labeled
import shutil

FLAGS = tf.flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#PATH/DIR parameters - INPUT
tf.flags.DEFINE_string("dataset","cifar10","which dataset")
#PATH/DIR parameters - INPUT (CIFAR10/SVHN only)
tf.flags.DEFINE_string("affmat_path","./affmat/cifar10/AE_LR=0.001/k=100/Affmat.tfrecords","path to affinity matrix for CIFAR10/SVHN")
tf.flags.DEFINE_string("dataset_dir","./dataset/cifar10/","folder of chosen dataset for CIFAR10/SVHN")
#PATH/DIR parameters - OUTPUT
tf.flags.DEFINE_string('log_dir', "./log/toy/classification/", "log_dir")
tf.flags.DEFINE_string('train_dir', './tmp/toy_classifier/',
                           """Directory where to write event logs """
"""and checkpoint.""")
#AFFMAT parameters
tf.flags.DEFINE_float('affmat_sigma', 100000, "sigma for the gaussian kernel of the affinity matrix")
tf.flags.DEFINE_integer('affmat_k', 15, "#neighbors for the gaussian kernel of the affinity matrix")


#LOSS funnction params
tf.flags.DEFINE_float('lgc_alpha', 0.9, "parameter for LGC algorithm")

tf.flags.DEFINE_string('loss_func', "lgc", "loss function")
tf.flags.DEFINE_float('learning_rate', 0.0001, "initial leanring rate")

#Seeds
tf.flags.DEFINE_integer('label_seed', 1, "seed used to select labels")
tf.flags.DEFINE_integer('seed', 1, "seed used to select labels")

#NUM LABELED
tf.flags.DEFINE_integer('num_labeled', 4000, "seed used to select labels")

#Batch size, etc
tf.flags.DEFINE_bool('validation', False, "")
tf.flags.DEFINE_integer('batch_size', 1000, "the number of examples in a batch")
tf.flags.DEFINE_integer('ul_batch_size', 4000, "the number of unlabeled examples in a batch")
tf.flags.DEFINE_integer('eval_freq', 5, "")
tf.flags.DEFINE_float('mom1', 0.5, "initial momentum rate")
tf.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
#Epochs
tf.flags.DEFINE_integer('num_epochs', 1000, "the number of epochs for training")
tf.flags.DEFINE_integer('epoch_decay_start', 10, "epoch of starting learning rate decay")
tf.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")

NUM_LABELED_EXAMPLES = 1000
NUM_EVAL_EXAMPLES = 5000

tf.logging.set_verbosity(tf.logging.ERROR)
FLAGS.log_dir = os.path.abspath(FLAGS.log_dir)
FLAGS.train_dir = os.path.abspath(FLAGS.train_dir)
if os.path.exists(FLAGS.log_dir):
    shutil.rmtree(FLAGS.log_dir)
os.mkdir(FLAGS.log_dir)
if os.path.exists(FLAGS.train_dir):
    shutil.rmtree(FLAGS.train_dir)
os.mkdir(FLAGS.train_dir)


def build_eval_graph(X_l,Y_l):
    #Todo: implement this
    losses = {}
    logit_l = sup_classifier.logit_small(X_l, is_training=False, update_batch_stats=False)
    
    losses['xent_sup'] = L.ce_loss(logit_l, Y_l)
    return losses

def build_training_graph(is_training,X_l,Y_l,ID_l,X_u,ID_u,K,lr,mom,lgc_alpha):
    CACHE_F = tf.get_variable("cache_f")
    W = tf.get_variable("Affinity_matrix")
    D = tf.get_variable("D")
    assert(K.shape[0] == W.shape[0])
    with tf.variable_scope("CNN",reuse=tf.AUTO_REUSE) as scope:
        losses = {}
        #logit_l = sup_classifier.logit_small(X_l,num_classes = Y_l.shape[1], is_training=is_training, update_batch_stats=False)
        #logit_u = sup_classifier.logit_small(X_u,num_classes = Y_l.shape[1], is_training=is_training, update_batch_stats=is_training)
    logit_l = tf.gather(CACHE_F,ID_l)
    logit_u = tf.gather(CACHE_F,ID_u)
    
    losses['xent_sup'] = L.ce_loss(logit_l, Y_l)
    losses['mse_sup'] = tf.losses.mean_squared_error(tf.nn.softmax(logit_l), Y_l)
    losses['mean_acc'] = L.accuracy(logit_l, Y_l)

    
    #Concatenate ids and logits
    ids = tf.concat([ID_l,ID_u], 0)
    logits = tf.concat([logit_l,logit_u], 0)
    logits = tf.nn.softmax(logits)
    #assert ids.shape[0] == logits.shape[0]
    
    if FLAGS.loss_func == "lgc":
        #Unsupervised loss
        K_ids = tf.gather(K, ids,axis=0)  #Get neighbor pairs
        K_ids_i = tf.reshape(K_ids[:,:,0],[-1])
        K_ids_j = tf.reshape(K_ids[:,:,1],[-1])
        logits_ids = tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(logits)[0]),-1),[1,FLAGS.affmat_k]),[-1]) 
        
         
        #all_F_i = tf.gather(logits,logits_ids,axis=0)
        all_F_i = tf.gather(tf.nn.softmax(CACHE_F),K_ids_i,axis=0)
        all_F_j = tf.gather(tf.nn.softmax(CACHE_F),K_ids_j,axis=0)  #F_j comes from cache
        all_Wij = tf.reshape(tf.gather(W,ids,axis=0),[-1])
        all_Dii = tf.gather(D,K_ids_i,axis=0)
        all_Djj = tf.gather(D,K_ids_j,axis=0)
        all_Dii = tf.tile(all_Dii[:,None],[1,all_F_i.shape[1]])
        all_Djj = tf.tile(all_Djj[:,None],[1,all_F_j.shape[1]])
        

        all_Fi = tf.multiply(all_Dii,all_F_i)
        all_Fj = tf.multiply(all_Djj,all_F_j)
        LGC_unsupervised_loss = (tf.multiply(tf.reduce_sum(tf.square(all_Fi - all_Fj),axis=1),all_Wij))
        #LGC_unsupervised_loss = tf.reduce_sum(tf.square(all_F_i - all_F_j),axis=1)
        losses["lgc_unsupervised_loss"] = tf.reduce_sum(LGC_unsupervised_loss)
        losses["lgc_supervised_loss"] = losses['xent_sup']
        
        losses["lgc_unsupervised_loss"] = (int(K.shape[0])/int(FLAGS.batch_size + FLAGS.ul_batch_size)) * losses["lgc_unsupervised_loss"]
        losses["lgc_supervised_loss"] = (FLAGS.num_labeled/int(FLAGS.batch_size))  * losses['mse_sup']
        
        
        lgc_lamb = 1/lgc_alpha - 1
        losses["lgc_loss"] =  losses["lgc_unsupervised_loss"] + lgc_lamb*losses["lgc_supervised_loss"]
                             
       

        
        #Assign to cache 
        assign_op_l = tf.scatter_update(ref=CACHE_F,indices=ID_l,updates=tf.nn.softmax(logit_l))
        assign_op_u = tf.scatter_update(ref=CACHE_F,indices=ID_u,updates=tf.nn.softmax(logit_u))
        #assign_to_cache = tf.group(assign_op_l,assign_op_u)
        assign_to_cache = tf.no_op()
    
        #Get Trainable vars
        tvars = tf.trainable_variables()
        tvars_W = list(filter(lambda x: np.char.find(x.name,"Affinity_matrix") != -1,tvars))
        tvars_D = list(filter(lambda x: np.char.find(x.name,"D") != -1,tvars))
        tvars_CNN = list(filter(lambda x: np.char.find(x.name,"CNN") != -1,tvars))
        tvars_CACHE = list(filter(lambda x: np.char.find(x.name,"cache") != -1,tvars))
        
        print([var.name for var in tvars_CACHE])
        
        
    
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
        grads_and_vars = opt.compute_gradients(losses['lgc_loss'], tvars_CACHE)
        train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())
    else:
        assign_to_cache = None
        opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
        grads_and_vars = opt.compute_gradients(losses['xent_sup'], tvars)
        train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())
    
    return losses, train_op, assign_to_cache, logit_l, all_Wij


        

def go(_):
    
    ds_dict = {}
    nxt_dict = {}
    it_dict = {}
    bs_dict = {"labeled":FLAGS.batch_size,"unlabeled":FLAGS.ul_batch_size,"test":FLAGS.ul_batch_size}
    size_dict = {"labeled":FLAGS.num_labeled}
    writer_dict = {}
    
    example_shape = None
    
    g1 = tf.Graph()
    

    with g1.as_default():
        
        with tf.device("/gpu:0"):
            with tf.variable_scope("Vars",reuse=tf.AUTO_REUSE):
                print("Reading dataset...")
                if FLAGS.dataset == "cifar10":
                    cifar_dict = load_whole_dataset(os.path.join(FLAGS.dataset_dir,"train.tfrecords"),
                                             get_images=False,
                                             get_zcas=True,
                                             get_labels=True,
                                             get_emb=False, ds_size=50000)
                    train_X = cifar_dict.pop("zca")
                    train_Y = cifar_dict.pop("label")
                    size_dict["X_shape"] = list(train_X[0,:].shape)
                    size_dict["Y_shape"] = list(train_Y[0,:].shape)
                    
                    cifar_dict = load_whole_dataset(os.path.join(FLAGS.dataset_dir,"test.tfrecords"),
                                             get_images=False,
                                             get_zcas=True,
                                             get_labels=True,
                                             get_emb=False, ds_size=10000)
                    test_X = cifar_dict.pop("zca")
                    test_Y = cifar_dict.pop("label")
                    ds_dict["test"] = tf.data.Dataset.from_tensor_slices({"X":test_X,"Y":test_Y})
                    size_dict["test"] = test_X.shape[0]          
                elif FLAGS.dataset == "svhn":
                    svhn_dict = load_whole_dataset(os.path.join(FLAGS.dataset_dir,"train.tfrecords"),
                             get_images=True,
                             get_zcas=False,
                             get_labels=True,
                             get_emb=False, ds_size=73257)
                    train_X = svhn_dict.pop("image")
                    train_Y = svhn_dict.pop("label")
                    
                    svhn_dict = load_whole_dataset(os.path.join(FLAGS.dataset_dir,"test.tfrecords"),
                             get_images=True,
                             get_zcas=False,
                             get_labels=True,
                             get_emb=False, ds_size=26032)
                    test_X = svhn_dict.pop("image")
                    test_Y = svhn_dict.pop("label")
                    ds_dict["test"] = tf.data.Dataset.from_tensor_slices({"X":test_X,"Y":test_Y})
                    size_dict["test"] = test_X.shape[0]
                
                if FLAGS.dataset == "cifar10" or FLAGS.dataset == "svhn":
                    
                    
                    #Form labeled and unlabeled datasets
                    perm = np.random.RandomState(seed=FLAGS.label_seed).permutation(np.arange(train_X.shape[0]))
                    train_X_l = train_X[perm[0:FLAGS.num_labeled],:]
                    train_Y_l = train_Y[perm[0:FLAGS.num_labeled],:]
                    train_X_ul = train_X[perm[FLAGS.num_labeled:],:]
                    train_Y_ul = train_Y[perm[FLAGS.num_labeled:],:]
                    del train_X
                    
                    init_conf_vals = gutils.init_matrix(np.argmax(train_Y,axis=1),perm[0:FLAGS.num_labeled])
                    CACHE_F = tf.get_variable(name="cache_f",initializer=tf.constant(init_conf_vals,dtype=tf.float32),trainable=True)
                    
                    size_dict["X_shape"] = list(train_X_l [0,:].shape)
                    size_dict["Y_shape"] = list(init_conf_vals[0,:].shape)
                    print(train_Y.shape)
                    
                    #Create variables for initial labels
                    PRIOR_Y_l = tf.Variable(init_conf_vals[perm[0:FLAGS.num_labeled],:],name="prior_yl")
                    
                    #Update dicts
                    ds_dict["labeled"] = tf.data.Dataset.from_tensor_slices({"X":train_X_l,"Y":train_Y_l,
                                                                          "ID":np.reshape(perm[0:FLAGS.num_labeled],[-1,1])})
                    ds_dict["unlabeled"] = tf.data.Dataset.from_tensor_slices({"X":train_X_ul,"Y":train_Y_ul,
                                                                          "ID":np.reshape(perm[FLAGS.num_labeled:],[-1,1])})
                    size_dict["unlabeled"] = train_X_ul.shape[0]   
                    print("Reading dataset...Done!")
                    
                    #Load affmat
                    print("Loading Affmat...")
                    K, AFF = load_whole_affmat(FLAGS.affmat_path, ds_size = 50000 if FLAGS.dataset == "cifar10" else 73257)
                    K = K[:,0:FLAGS.affmat_k,:]
                    AFF = AFF[:,0:FLAGS.affmat_k]
                    K = tf.constant(K,dtype=tf.int64)
                    AFF = tf.exp(-np.square(AFF)/(2*FLAGS.affmat_sigma*FLAGS.affmat_sigma))
                    AFF = tf.get_variable(name="Affinity_matrix",initializer=tf.cast(AFF,dtype=tf.float32))
                    print("Loading Affmat...Done!")                                                       
                else:
                    #Extract info for toy_dict
                    toy_dict = toy.getTFDataset(FLAGS.dataset, FLAGS.num_labeled, FLAGS.label_seed)            
                    df_x = toy_dict.pop("df_x") #Used to create Affinity Mat and infer num_unlabeled.  
                    df_y_l = toy_dict.pop("df_y_l") #Used to create var
                    df_y = toy_dict.pop("df_y") #Used to create var
                    
                    
                    init_conf_vals = toy_dict.pop("INIT")#Used to infer Y shape             
                    ds_dict = {"labeled": toy_dict["labeled"],
                               "unlabeled" : toy_dict["unlabeled"]}

                    
                    
                    #Create variable for initial labels and F cache
                    PRIOR_Y_l = tf.Variable(df_y_l,name="prior_yl")
                    CACHE_F = tf.get_variable(name="cache_f",initializer=init_conf_vals,trainable=True)
                    
                    #Update dicts
                    size_dict["X_shape"] = list(df_x [0,:].shape)
                    size_dict["Y_shape"] = list(init_conf_vals[0,:].shape)
                    size_dict["unlabeled"] = df_x.shape[0] - FLAGS.num_labeled
                    print("Reading dataset...Done!")
                    
                    #Load affmat
                    print("Loading Affmat...")
                    W= AffMatGenerator(dist_func="gaussian",mask_func="knn",k=FLAGS.affmat_k,sigma=FLAGS.affmat_sigma).\
                                                generateAffMat(df_x)
                    #Convert K to [:,0:K,2] array, i.e. 2D array of [i,j] pairs
                    K = np.zeros((W.shape[0],FLAGS.affmat_k,2),dtype=np.int64)
                    for i in np.arange(W.shape[0]):
                        K[i,:,0] = i
                    K[:,:,1] = np.argsort(-W,axis=1)[:,:FLAGS.affmat_k]
                    #Create corresponding [:,0:K] tensor containing the distances
                    AFF = np.zeros((W.shape[0],FLAGS.affmat_k))
                    for i in np.arange(W.shape[0]):
                        for j in np.arange(FLAGS.affmat_k):
                            AFF[i,j] = W[K[i,j,0],K[i,j,1]]
                            assert(AFF[i,j]>0)
       
                    AFF = tf.get_variable(name="Affinity_matrix",initializer=tf.cast(AFF,dtype=tf.float32))
                    
                    print("Loading Affmat...Done!")            
                
                #Create D variable
                D = tf.get_variable(name="D",dtype=tf.float32,
                                    initializer=tf.math.reduce_sum(
                                        tf.get_variable("Affinity_matrix"),axis=1)\
                                    )

                
                size_dict["train"] = size_dict["labeled"] + size_dict["unlabeled"]
                
                
                #Make the datasets shuffle,repeat,batch
                ds_dict["train_eval"] = ds_dict["labeled"].concatenate(ds_dict["unlabeled"]).batch(FLAGS.ul_batch_size)
                ds_dict["unlabeled_eval"] = ds_dict["unlabeled"].batch(FLAGS.batch_size)
                ds_dict["labeled_eval"] = ds_dict["labeled"].batch(FLAGS.ul_batch_size)
                ds_dict["unlabeled"] = ds_dict["unlabeled"].shuffle(1000).repeat().batch(FLAGS.ul_batch_size)
                ds_dict["labeled"] = ds_dict["labeled"].shuffle(1000).repeat().batch(FLAGS.batch_size)
                if "test" in ds_dict.keys():
                    ds_dict["test_eval"] = ds_dict["test"].batch(FLAGS.batch_size)
                    ds_dict["test"] = ds_dict["test"].shuffle(size_dict["test"]).repeat().batch(FLAGS.ul_batch_size)
                    
                #Create variable for initial unlabeled
                TRAIN_Y_UL = tf.Variable(np.zeros((size_dict["unlabeled"],size_dict["Y_shape"][0])),name="y_ul")
                
                
                for key, value in ds_dict.items():
                    it_dict[key] = value.make_initializable_iterator()
                    nxt_dict[key] = it_dict[key].get_next()
                
              
                
                    
        with tf.device("/gpu:0"):  
            
            with tf.variable_scope("Vars",reuse=tf.AUTO_REUSE):
                #Define config
                tfconfig = tf.ConfigProto(allow_soft_placement=True)
                tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.7
    
                with tf.Session(config=tfconfig) as sess:

                    
                    print("Setting placeholders...")
                    #PLACEHOLDERS
                    LGC_ALPHA = tf.placeholder_with_default(tf.cast(0.00001,tf.float32), shape=[], name="learning_rate")
                    
                    lr = tf.placeholder_with_default(tf.cast(FLAGS.learning_rate,tf.float32), shape=[], name="learning_rate")
                    mom = tf.placeholder_with_default(tf.cast(FLAGS.mom1,tf.float32), shape=[], name="momentum")
                    X_l = tf.placeholder(dtype=tf.float32,shape=[None]+size_dict["X_shape"],name="placeholder/X_l")
                    Y_l = tf.placeholder(dtype=tf.float32,shape=[None]+size_dict["Y_shape"],name="placeholder/Y_l")
                    X_u = tf.placeholder(dtype=tf.float32,shape=[None]+size_dict["X_shape"],name="placeholder/X_ul")
                    Y_u = tf.placeholder(dtype=tf.float32,shape=[None]+size_dict["Y_shape"],name="placeholder/Y_ul")
                    ID_l = tf.placeholder(dtype=tf.int64,shape=[None],name="placeholder/ID_l")
                    ID_u = tf.placeholder(dtype=tf.int64,shape=[None],name="placeholder/ID_u")
                    
                    print("Setting placeholders...Done!")
                   
                
                    
                    
                    
                    print("Setting writers...")
                    #Create FileWriter
                    if not FLAGS.log_dir:
                        writer_dict["train"] = None
                        writer_dict["labeled"] = None
                        writer_dict["unlabeled"] = None
                        writer_dict["test"] = None
                    else:
                        writer_dict["train"] = tf.summary.FileWriter(FLAGS.log_dir + "/train")
                        writer_dict["labeled"] = tf.summary.FileWriter(FLAGS.log_dir + "/labeled")
                        writer_dict["unlabeled"] = tf.summary.FileWriter(FLAGS.log_dir + "/unlabeled")
                        writer_dict["test"] = tf.summary.FileWriter(FLAGS.log_dir + "/test")     
                    print("Setting writers...Done!")
                    
                    print("Setting training graph...")
                    #Build training_graph
                    loss, train_op, cache_op,_,extra  = build_training_graph(is_training=True,
                                                     X_l=X_l,Y_l=Y_l,
                                                     X_u=X_u,ID_l=ID_l,
                                                     ID_u = ID_u,
                                                     K=K,lr=lr, mom=mom,
                                                     lgc_alpha= LGC_ALPHA)
                    
                    print("Setting training graph...Done!")
                    
                    print("Setting test graph...")
                    # Build eval graph
                    eval_loss, _, _,eval_logit,_  = build_training_graph(is_training=False,
                                                     X_l=X_l,Y_l=Y_l,
                                                     X_u=X_u,ID_l=ID_l,
                                                     ID_u = ID_u,
                                                     K=K,lr=lr, mom=mom,
                                                     lgc_alpha= LGC_ALPHA)
                    
                    print("Setting test graph... Done!")
                    print((tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
                    print((tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)))
                          
                    
    #####################################################################################
                    mon_sess = sess                
                    
                    for var in tf.global_variables():
                        print("Initializing Variable {}: shape {}".format(var.name,var.shape,flush=True))
                        mon_sess.run(var.initializer)
                        #print(mon_sess.run(var))
                        print("Initializing Variable {}: shape {}...Done!".format(var.name,var.shape))
                    '''
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, FLAGS.train_dir)
                    print("Model saved in path: %s" % save_path)
                    '''

                
                    for ep in range(FLAGS.num_epochs):
                        print("EPOCH:{}".format(ep))
                        
                        #Adjust decay if necessary
                        if ep < FLAGS.epoch_decay_start:
                            feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1,
                                         LGC_ALPHA:FLAGS.lgc_alpha}
                            print("MOMENTUM:{},lr:".format(FLAGS.mom1,FLAGS.lgc_alpha))
                        else:
                            decayed_lr = ((FLAGS.num_epochs - ep) / float(
                                FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                            feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1,
                                         LGC_ALPHA:FLAGS.lgc_alpha}
                        
                        #Initialize loss,time and iterator
                        start = time.time()
                        mon_sess.run([it_dict["labeled"].initializer,it_dict["unlabeled"].initializer])
                        losses_dict = {}                 
                        for k in loss.keys():
                            losses_dict[k] = 0.0
                        #Run training examples
                        for i in range(FLAGS.num_iter_per_epoch):
                            nxt_l, nxt_u = mon_sess.run([nxt_dict["labeled"],nxt_dict["unlabeled"]])
                            feed_dict[X_l] = nxt_l["X"]
                            feed_dict[Y_l] = nxt_l["Y"]
                            
                            feed_dict[ID_l] = np.reshape(nxt_l["ID"],[-1])
                            
                            feed_dict[X_u] = nxt_u["X"]
                            #feed_dict[Y_u] = nxt_u["Y"]
                            feed_dict[ID_u] = np.reshape(nxt_u["ID"],[-1])
                            if ep < FLAGS.epoch_decay_start:
                                _, batch_loss  = mon_sess.run([train_op, loss],
                                                        feed_dict=feed_dict)
                                
                            else:
                                _, _,batch_loss  = mon_sess.run([train_op, cache_op, loss],
                                                        feed_dict=feed_dict)
                            for k,v in batch_loss.items():
                                losses_dict[k] += v
                            
                        #Print elapsed time, get global step
                        end = time.time()
                        current_global_step = tf.train.get_global_step().eval(mon_sess)
                        
                        #Get mean of losses
                        for k,v in batch_loss.items():
                            losses_dict[k] /= FLAGS.num_iter_per_epoch

                        #Add Summary
                        summary = tf.Summary()
                        for k,v in batch_loss.items():
                            summary.value.add(tag=k, simple_value=v)
                        writer_dict["train"].add_summary(summary, current_global_step)
                        
                        print("Epoch:", ep,
                              "; LGC loss",losses_dict["lgc_loss"],
                              "; LGC_sup_loss",losses_dict["lgc_supervised_loss"],
                              "; LGC_unsup_loss",losses_dict["lgc_unsupervised_loss"],
                              "; sup_acc",losses_dict["mean_acc"],
                              
                               "; elapsed_time:", end - start)
                    


                        
                        ''' EVAL Procedure '''
                        if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs: 
                            
                            def eval(KEY,is_writing=True):
                                sum_loss = 0
                                start = time.time()
                                mon_sess.run(it_dict[KEY + "_eval"].initializer)
                                pred_Y = None
                                actual_Y = None
                                IDs = None
                                #Run eval examples
                                while True:
                                    try:
                                        nxt = mon_sess.run(nxt_dict[KEY+"_eval"])
                                        feed_dict[X_l] = nxt["X"]
                                        feed_dict[Y_l] = nxt["Y"]
                                        feed_dict[ID_l] = np.reshape(nxt["ID"],[-1])
                                        mean_acc, logit_l  = mon_sess.run([eval_loss["mean_acc"],eval_logit],
                                                                    feed_dict=feed_dict)
                                        sum_loss += mean_acc  * nxt["X"].shape[0]
                                        if pred_Y is None:
                                            pred_Y = logit_l
                                            actual_Y = nxt["Y"]
                                            IDs = nxt["ID"]
                                        else:
                                            pred_Y = np.concatenate([pred_Y,logit_l])
                                            actual_Y = np.concatenate([actual_Y,nxt["Y"]])
                                            IDs = np.concatenate([IDs,nxt["ID"]])
                                        
                                        
                                    except tf.errors.OutOfRangeError:
                                        break
                                #Print elapsed time, get global step
                                end = time.time()
                                current_global_step = tf.train.get_global_step().eval(mon_sess)
                                

                                #Add Summary
                                summary = tf.Summary()
                                summary.value.add(tag="acc", simple_value=sum_loss/size_dict[KEY])
                                if is_writing:
                                    writer_dict[KEY].add_summary(summary, current_global_step)
                                print("Eval {}: {} accuracy ".format(KEY,sum_loss/size_dict[KEY]))
                                
                                #Sort pred w.r.t ids
                                IDs = np.reshape(IDs,[-1]).tolist()
                                return(pred_Y,actual_Y,IDs)
                            
                            eval("labeled")
                            eval("unlabeled")
                            #if "test" in ds_dict.keys():
                            #    eval("test")
                            
                            if not FLAGS.dataset in ["svhn","cifar10"]:
                                pred_Y, actual_Y, pred_ids = eval("train")
                                
                                pred_Y[pred_ids,:] = pred_Y
                                actual_Y[pred_ids,:] = actual_Y
                                
                                
                                pred_Y = np.argmax(mon_sess.run(CACHE_F),axis=1)
                                actual_Y = np.argmax(actual_Y,axis=1)
                                
                                labeledIndexes = np.zeros([pred_Y.shape[0]],dtype=np.bool)
                                labeledIndexes[pred_ids[0:FLAGS.num_labeled]] = True
                                
                                if (ep + 1) == FLAGS.eval_freq :
                                    vertex_opt = sslplot.vertexplotOpt(pred_Y,size=5,\
                                                                       labeledIndexes=labeledIndexes)
                                    sslplot.plotGraph(df_x, W, vertex_opt, online=False, interactive=False,
                                                        title="NN pred - labeled",
                                                        plot_filename="0.png")
                                vertex_opt = sslplot.vertexplotOpt(pred_Y,size=5,\
                                                                   labeledIndexes=np.logical_not(labeledIndexes))
                                sslplot.plotGraph(df_x, W, vertex_opt, online=False, interactive=False,
                                                   title="NN pred - unlabeled",plot_filename=str(current_global_step)+".png")
                            
                            

                            
                            
                    
 
def runSetOfExperiments(_):
    go(None)


if __name__ == '__main__':
    tf.app.run()