import argparse
import os
import tensorflow as tf
from model import GraphSSL


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("dataset","spiral_hard","chosen dataset")
tf.flags.DEFINE_float("split_test",10/1500,"Percent of dataset to be used for testing")
tf.flags.DEFINE_bool("can_plot",True,"enable plotting ops")


def main(_):

    with tf.device("/gpu:0"):
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        #tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=tfconfig) as sess:
            model = GraphSSL(sess, FLAGS)
            model.build_model()
            model.train()
            ######model.train(args) if args.phase == 'train' \
            ######    else model.test(args)
        
if __name__ == '__main__':
    tf.app.run()