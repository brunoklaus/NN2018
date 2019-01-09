import argparse
import os
import tensorflow as tf
from graph_ssl.model import GraphSSL

parser = argparse.ArgumentParser(description='')


parser.add_argument('--dataset', dest='dataset', default='spiral_hard', help='chosen dataset')



args = parser.parse_args()  


def main(_):
    '''if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    '''

    with tf.device("/gpu:0"):
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        #tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.1
        with tf.Session(config=tfconfig) as sess:
            model = GraphSSL(sess, args)
            model.build_model()
            model.train()
            ######model.train(args) if args.phase == 'train' \
            ######    else model.test(args)
        
if __name__ == '__main__':
    tf.app.run()