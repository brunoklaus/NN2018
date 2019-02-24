import argparse
import os
import tensorflow as tf
from nn.cifar10 import  inputs
from nn import sup_classifier
import nn.layers as L

from datetime import datetime
import time
from progressbar import progressbar
from python_utils.time import epoch

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset","cifar10","chosen dataset")
tf.flags.DEFINE_float("split_test",10/1500,"Percent of dataset to be used for testing")
tf.flags.DEFINE_bool("can_plot",True,"enable plotting ops")



tf.flags.DEFINE_string('log_dir', "../log/cifar10/", "log_dir")
tf.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.flags.DEFINE_bool('validation', False, "")

tf.flags.DEFINE_integer('batch_size', 32, "the number of examples in a batch")
tf.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.flags.DEFINE_integer('eval_batch_size', 100, "the number of eval examples in a batch")
tf.flags.DEFINE_integer('eval_freq', 5, "")
tf.flags.DEFINE_integer('num_epochs', 120, "the number of epochs for training")
tf.flags.DEFINE_integer('epoch_decay_start', 80, "epoch of starting learning rate decay")
tf.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.flags.DEFINE_float('learning_rate', 0.001, "initial leanring rate")
tf.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")



tf.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.flags.DEFINE_integer('log_frequency', 10,
"""How often to log results to the console.""")
tf.flags.DEFINE_string('train_dir', '../tmp/cifar10_train',
                           """Directory where to write event logs """
"""and checkpoint.""")



NUM_EVAL_EXAMPLES = 5000

tf.logging.set_verbosity(tf.logging.ERROR)

def build_eval_graph(x, y, ul_x):
    losses = {}
    logit = sup_classifier.logit(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    return losses

def build_training_graph(x, y, ul_x, lr, mom):
    logit = sup_classifier.logit(x, is_training=True,update_batch_stats=True, stochastic=True)
    nll_loss = L.ce_loss(logit, y)
    loss = nll_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())
    return loss, train_op

def main(_):
    
    g1 = tf.Graph()
    

    with g1.as_default():
        with tf.device("/cpu:0"):
                    images, labels = inputs(batch_size=FLAGS.batch_size,
                                            train=True,
                                            validation=FLAGS.validation,
                                            shuffle=True)
        
                    ul_images = unlabeled_inputs(batch_size=FLAGS.ul_batch_size,
                                                 validation=FLAGS.validation,
                                                 shuffle=True)
        
                    images_eval_train, labels_eval_train = inputs(batch_size=FLAGS.eval_batch_size,
                                                                  train=True,
                                                                  validation=FLAGS.validation,
                                                                  shuffle=True)
                    ul_images_eval_train = unlabeled_inputs(batch_size=FLAGS.eval_batch_size,
                                                            validation=FLAGS.validation,
                                                            shuffle=True)
        
                    images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
                                                                train=False,
                                                                validation=FLAGS.validation,
                                                                shuffle=True)
         
   
        with tf.device("/gpu:0"):  
            #Define config
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5     
            
            #BUILD TRAINING GRAPH
            lr = tf.placeholder_with_default(tf.cast(FLAGS.learning_rate,tf.float32), shape=[], name="learning_rate")
            mom = tf.placeholder_with_default(tf.cast(FLAGS.mom1,tf.float32), shape=[], name="momentum")
            with tf.variable_scope("CNN") as scope:
                    loss, train_op  = build_training_graph(images, labels, ul_images, lr, mom)
                    scope.reuse_variables()
                    # Build eval graph
                    losses_eval_train = build_eval_graph(images_eval_train, labels_eval_train, ul_images_eval_train)
                    losses_eval_test = build_eval_graph(images_eval_test, labels_eval_test, images_eval_test)
            
            #Crete FileWriter
            if not FLAGS.log_dir:
                logdir = None
                writer_train = None
                writer_test = None
            else:
                logdir = FLAGS.log_dir
                writer_train = tf.summary.FileWriter(FLAGS.log_dir + "/train", g1)
                writer_test = tf.summary.FileWriter(FLAGS.log_dir + "/test", g1)
            
            with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            hooks=[tf.train.NanTensorHook(loss),],
            config=tfconfig) as mon_sess:
                for ep in range(FLAGS.num_epochs):
                    print("EPOCH:{}".format(ep))
                    if ep < FLAGS.epoch_decay_start:
                        feed_dict = {lr: FLAGS.learning_rate, mom: FLAGS.mom1}
                        print("MOMENTUM:{},lr:".format(FLAGS.mom1,FLAGS.learning_rate))
                    else:
                        decayed_lr = ((FLAGS.num_epochs - ep) / float(
                            FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                        feed_dict = {lr: decayed_lr, mom: FLAGS.mom2}
    
                    sum_loss = 0
                    start = time.time()
                    for i in range(FLAGS.num_iter_per_epoch):
                        _, batch_loss  = mon_sess.run([train_op, loss],
                                                    feed_dict=feed_dict)
                        sum_loss += batch_loss
                    end = time.time()
                    print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch, "elapsed_time:", end - start)
                    
                    '''BEGIN EVAL'''
                    
                    if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs:
                        # Eval on training data
                        act_values_dict = {}
                        for key, _ in losses_eval_train.items():
                            act_values_dict[key] = 0
                        n_iter_per_epoch = NUM_EVAL_EXAMPLES // FLAGS.eval_batch_size
                        for i in range(n_iter_per_epoch):
                            values = list(losses_eval_train.values())
                            act_values = mon_sess.run(values)
                            for key, value in zip(act_values_dict.keys(), act_values):
                                act_values_dict[key] += value
                        summary = tf.Summary()
                        current_global_step = tf.train.get_global_step().eval(mon_sess)
                        for key, value in act_values_dict.items():
                            print("train-" + key, value / n_iter_per_epoch)
                            summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                        if writer_train is not None:
                            writer_train.add_summary(summary, current_global_step)
    
                        # Eval on test data
                        act_values_dict = {}
                        for key, _ in losses_eval_test.items():
                            act_values_dict[key] = 0
                        n_iter_per_epoch = NUM_EVAL_EXAMPLES // FLAGS.eval_batch_size
                        for i in range(n_iter_per_epoch):
                            values = list(losses_eval_test.values())
                            act_values = mon_sess.run(values)
                            for key, value in zip(act_values_dict.keys(), act_values):
                                act_values_dict[key] += value
                        summary = tf.Summary()
                        current_global_step = tf.train.get_global_step().eval(mon_sess)
                        for key, value in act_values_dict.items():
                            print("test-" + key, value / n_iter_per_epoch)
                            summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                        if writer_test is not None:
                            writer_test.add_summary(summary, current_global_step)
                    ''' END EVAL'''
 
if __name__ == '__main__':
    tf.app.run()
