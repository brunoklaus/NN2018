
import argparse
import os
import tensorflow as tf
from dataset_utils import convert_images_and_labels_and_emb
import sup_classifier
import layers as L
import numpy as np
from datetime import datetime
import time
from progressbar import progressbar
from python_utils.time import epoch
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset","svhn","chosen dataset")

tf.flags.DEFINE_string('autoencoder_mode', "embedding", "whether to train or save tfrecords")
tf.flags.DEFINE_bool('use_zca', False, "whether to use ZCA or not")
tf.flags.DEFINE_string('log_dir', "../log/svhn/autoencoder/", "log_dir")
tf.flags.DEFINE_string('train_dir', '../tmp/svhn_autoencoder_2',
                           """Directory where to write event logs """
"""and checkpoint.""")
tf.flags.DEFINE_string('emb_name', 'AE_embedding',"name of embedding to be saved")
tf.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")


tf.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.flags.DEFINE_bool('validation', False, "")
tf.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
tf.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.flags.DEFINE_integer('eval_batch_size', 100, "the number of eval examples in a batch")
tf.flags.DEFINE_integer('eval_freq', 5, "")
tf.flags.DEFINE_integer('num_epochs', 120, "the number of epochs for training")
tf.flags.DEFINE_integer('epoch_decay_start', 20, "epoch of starting learning rate decay")
tf.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")

tf.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")


if FLAGS.dataset == "cifar10":
    from cifar10 import  inputs, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST
elif FLAGS.dataset == "svhn":
    from svhn import  inputs, NUM_EXAMPLES_TRAIN, NUM_EXAMPLES_TEST
else:
    raise ValueError("Unknown dataset")

NUM_EVAL_EXAMPLES = 5000

tf.logging.set_verbosity(tf.logging.ERROR)

def build_eval_graph(IMG,ZCA):
    losses = {}
    
    loss, _, sample = sup_classifier.autoencoder(IMG,ZCA, is_training=False, update_batch_stats=False,
                                                            use_zca=FLAGS.use_zca)
    losses['reconstruction_loss'] = loss
    return losses, sample

def build_training_graph(IMG,ZCA,lr, mom):
    loss, _,_ = sup_classifier.autoencoder(IMG,ZCA, is_training=True,update_batch_stats=True, stochastic=True,
                                           use_zca=FLAGS.use_zca)

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())
    return loss, train_op

def build_emb_graph(IMG,ZCA):    
    _, latent_space, _ = sup_classifier.autoencoder(IMG,ZCA, is_training=False, update_batch_stats=False,
                                                     use_zca=FLAGS.use_zca)
    return latent_space

def read_images():
    D = inputs(batch_size=FLAGS.ul_batch_size)
    inp_D = dict()
    inp_init = dict()
    inp_next = dict()
    
    inp_D["AE_train"] = D["train"].shuffle(buffer_size=1000).repeat().batch(FLAGS.ul_batch_size)
    inp_D["AE_eval_train"] = D["train"].shuffle(buffer_size=1000).take(NUM_EVAL_EXAMPLES).batch(FLAGS.eval_batch_size)
    inp_D["AE_eval_test"] = D["test"].shuffle(buffer_size=1000).take(NUM_EVAL_EXAMPLES).batch(FLAGS.eval_batch_size)
    inp_D["AE_emb"] = D["train"].batch(FLAGS.eval_batch_size)
    
    
    for k in inp_D.keys():
        inp_init[k] = inp_D[k].make_initializable_iterator()
        inp_next[k] = inp_init[k].get_next()

    return inp_D, inp_init, inp_next

def classifier_eval(mon_sess,losses_eval,initializer,nxt_op,IMG,ZCA,writer,description="train"):
    #BEGIN: Eval
    act_values_dict = {}
    for key, _ in losses_eval.items():
        act_values_dict[key] = 0
    n_iter_per_epoch = NUM_EVAL_EXAMPLES // FLAGS.eval_batch_size
     
     
    mon_sess.run(initializer)
    for i in range(n_iter_per_epoch):
        nxt = mon_sess.run(nxt_op)
         
        values = list(losses_eval.values())
        act_values = mon_sess.run(values,
                       feed_dict={IMG:nxt["image"],
                                  ZCA:nxt["zca"]})
        #Add to each loss
        for key, value in zip(act_values_dict.keys(), act_values):
            act_values_dict[key] += value
            
    #Create summary
    summary = tf.Summary()
    current_global_step = tf.train.get_global_step().eval(mon_sess)
    for key, value in act_values_dict.items():
        print(description + "-" +  key, value / n_iter_per_epoch)
        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
    if writer is not None:
        writer.add_summary(summary, current_global_step)
        #END: Eval
def autoencoder_run(_):
    
    g1 = tf.Graph()
    

    with g1.as_default():
        with tf.device("/cpu:0"):
            ds_dict, init_dict, nxt_dict = read_images()

   
        with tf.device("/gpu:0"):  
            #Define config
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5     
            
            #PLACEHOLDERS
            lr = tf.placeholder_with_default(tf.cast(FLAGS.learning_rate,tf.float32), shape=[], name="learning_rate")
            mom = tf.placeholder_with_default(tf.cast(FLAGS.mom1,tf.float32), shape=[], name="momentum")
            IMG = tf.placeholder(dtype=tf.float32,shape= (None,32,32,3),name="placeholder/IMG")
            ZCA = tf.placeholder(dtype=tf.float32,shape= (None,32,32,3),name="placeholder/ZCA")
            
            #BUILD GRAPH
            with tf.variable_scope("CNN") as scope:
                    loss, train_op  = build_training_graph(IMG,ZCA,lr, mom)
                    scope.reuse_variables()
                    # Build eval graph
                    losses_eval_train, sample = build_eval_graph(IMG,ZCA)
                    losses_eval_test, _ = build_eval_graph(IMG,ZCA)
                    latent_space = build_emb_graph(IMG, ZCA)
                    X = tf.placeholder(tf.float32, shape=[None, 128, 128, 3], name="X")
                    latent_space = tf.reshape(tensor=latent_space,shape=[FLAGS.eval_batch_size,-1])
                    
                   
            
            #Create FileWriter
            if not FLAGS.log_dir:
                writer_train = None
                writer_test = None
            else:
                writer_train = tf.summary.FileWriter(FLAGS.log_dir + "/train", g1)
                writer_test = tf.summary.FileWriter(FLAGS.log_dir + "/test", g1)
            #FileWriter for embedding
            if FLAGS.autoencoder_mode=="embedding":
                writer_emb =  tf.summary.FileWriter(FLAGS.log_dir + "emb", g1)
            else:
                writer_emb = None
                        
                        
                        
            with tf.train.MonitoredTrainingSession(\
            checkpoint_dir=FLAGS.train_dir,
            hooks=[],
            config=tfconfig) as mon_sess:
            
                if FLAGS.autoencoder_mode == "embedding":
                    
                    #Get embedded rep
                    mon_sess.run(init_dict["AE_emb"].initializer)
                    bs = FLAGS.eval_batch_size
                    
                    labels = np.zeros((NUM_EXAMPLES_TRAIN))
                    zca = np.zeros((NUM_EXAMPLES_TRAIN,32*32*3))
                    image = np.zeros((NUM_EXAMPLES_TRAIN,32*32*3))
                    
                    
                    
                    for i in range(NUM_EXAMPLES_TRAIN//bs):
                        nxt = mon_sess.run(nxt_dict["AE_emb"])
                        assert (nxt["id"][0] == i * bs)
                        labels[i * bs : (i+1)*bs] = np.reshape(np.argmax(nxt["label"],axis=1),(-1))
                        zca[i * bs : (i+1)*bs,:] = np.reshape(nxt["zca"],(bs,-1))
                        image[i * bs : (i+1)*bs,:] = np.reshape(nxt["image"],(bs,-1))
                        feed_dict = { IMG:nxt["image"],
                                      ZCA:nxt["zca"]}
                        emb = mon_sess.run(latent_space,feed_dict=feed_dict)
                        if i == 0:
                            all_emb = np.zeros((NUM_EXAMPLES_TRAIN,emb.shape[1]))
                        all_emb[i * bs : (i+1)*bs,:] = emb
                    
                    
                    convert_images_and_labels_and_emb(image, labels, zca, all_emb, os.path.join(FLAGS.data_dir, FLAGS.emb_name+'.tfrecords'))
                    
                    return(True)
            
            
                for ep in range(FLAGS.num_epochs):
                    print("EPOCH:{}".format(ep))
                    
                    #Adjust decay if necessary
                    if ep < FLAGS.epoch_decay_start:
                        feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                        print("MOMENTUM:{},lr:".format(FLAGS.mom1,FLAGS.learning_rate))
                    else:
                        decayed_lr = ((FLAGS.num_epochs - ep) / float(
                            FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                        feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}
                    #Initialize loss,time and iterator
                    sum_loss = 0
                    start = time.time()
                    mon_sess.run(init_dict["AE_train"].initializer)
                    #Run training examples
                    for i in range(FLAGS.num_iter_per_epoch):
                        nxt = mon_sess.run(nxt_dict["AE_train"])
                        feed_dict[IMG] = nxt["image"]
                        feed_dict[ZCA] = nxt["zca"]
                        _, batch_loss  = mon_sess.run([train_op, loss],
                                                    feed_dict=feed_dict)
                        sum_loss += batch_loss
                    #Print elapsed time
                    end = time.time()
                    print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch,
                           "elapsed_time:", end - start)

                    
                    ''' EVAL Procedure '''
                    if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs: 
                         
                        #BEGIN: Get Sample
                        sample_img  = mon_sess.run([sample],
                                                        feed_dict=feed_dict)
                        with tf.Graph().as_default():
                            tf.Session().run([tf.write_file("sample.png",tf.image.encode_png(sample_img[0]))])
                            print("saved sample")
                        #END: Get Sample   
                        #EVAL TRAIN
                        classifier_eval(initializer = init_dict["AE_eval_train"].initializer,
                                         nxt_op = nxt_dict["AE_eval_train"],
                                         losses_eval = losses_eval_train,
                                         writer=writer_train,
                                         mon_sess = mon_sess,
                                         IMG=IMG,ZCA=ZCA,description="train")
                        #EVAL TEST
                        classifier_eval(initializer = init_dict["AE_eval_test"].initializer,
                                     nxt_op = nxt_dict["AE_eval_test"],
                                     losses_eval = losses_eval_test,
                                     writer=writer_test,
                                     mon_sess = mon_sess,
                                     IMG=IMG,ZCA=ZCA,description="test")
                        
                        #END Eval Test data
                    ''' END EVAL'''
                        
def main(_):
    autoencoder_run(None)


if __name__ == '__main__':
    tf.app.run()
