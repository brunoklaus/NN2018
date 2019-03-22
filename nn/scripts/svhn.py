from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from scipy.io import loadmat

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf
from dataset_utils import *

DATA_URL_TRAIN = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
DATA_URL_TEST = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../dataset/svhn', "")
tf.app.flags.DEFINE_integer('num_labeled_examples', 1000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "The number of validation examples")
tf.app.flags.DEFINE_integer('dataset_seed', 1, "dataset seed")

NUM_EXAMPLES_TRAIN = 73257
NUM_EXAMPLES_TEST = 26032


def maybe_download_and_extract():
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    filepath_train_mat = os.path.join(FLAGS.data_dir, 'train_32x32.mat')
    filepath_test_mat = os.path.join(FLAGS.data_dir, 'test_32x32.mat')
    if not os.path.exists(filepath_train_mat) or not os.path.exists(filepath_test_mat):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        urllib.request.urlretrieve(DATA_URL_TRAIN, filepath_train_mat, _progress)
        urllib.request.urlretrieve(DATA_URL_TEST, filepath_test_mat, _progress)

    # Training set
    print("Loading training data...")
    print("Preprocessing training data...")
    train_data = loadmat(FLAGS.data_dir + '/train_32x32.mat')
    train_x = (-127.5 + train_data['X']) / 255.
    train_x = train_x.transpose((3, 0, 1, 2))
    train_x = train_x.reshape([train_x.shape[0], -1])
    train_y = train_data['y'].flatten().astype(np.int32)
    train_y[train_y == 10] = 0

    # Test set
    print("Loading test data...")
    test_data = loadmat(FLAGS.data_dir + '/test_32x32.mat')
    test_x = (-127.5 + test_data['X']) / 255.
    test_x = test_x.transpose((3, 0, 1, 2))
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = test_data['y'].flatten().astype(np.int32)
    test_y[test_y == 10] = 0

    np.save('{}/train_images'.format(FLAGS.data_dir), train_x)
    np.save('{}/train_labels'.format(FLAGS.data_dir), train_y)
    np.save('{}/test_images'.format(FLAGS.data_dir), test_x)
    np.save('{}/test_labels'.format(FLAGS.data_dir), test_y)


def load_svhn():
    maybe_download_and_extract()
    train_images = np.load('{}/train_images.npy'.format(FLAGS.data_dir)).astype(np.float32)
    train_labels = np.load('{}/train_labels.npy'.format(FLAGS.data_dir)).astype(np.float32)
    test_images = np.load('{}/test_images.npy'.format(FLAGS.data_dir)).astype(np.float32)
    test_labels = np.load('{}/test_labels.npy'.format(FLAGS.data_dir)).astype(np.float32)
    print(train_images.shape)
    
    print("Apply ZCA whitening")
    components, mean, train_zca = ZCA(train_images)
    print("var:{}".format(np.var(train_images)))
    print("mean:{}".format(np.mean(train_images)))
    np.save('{}/components'.format(FLAGS.data_dir), components)
    np.save('{}/mean'.format(FLAGS.data_dir), mean)
    test_zca = np.dot(test_images - mean, components.T)
    
    return (train_images,train_zca,train_labels), (test_images,test_zca,test_labels)


def prepare_dataset():
    (train_images, train_zca, train_labels), (test_images, test_zca, test_labels) = load_svhn()
    dirpath = os.path.join(FLAGS.data_dir)
    convert_images_and_labels(train_images,
                              train_labels,
                              train_zca,
                              os.path.join(dirpath, 'train.tfrecords'))
    
    convert_images_and_labels(test_images,
                              test_labels,
                              test_zca,
                              os.path.join(dirpath, 'test.tfrecords'))

 



def inputs(batch_size,label_seed=1,val_seed=1):
    dirpath = os.path.join(FLAGS.data_dir)
    dataset_train = tf.data.TFRecordDataset([os.path.join(dirpath, 'train.tfrecords')]).map(extract_fn)
    dataset_test = tf.data.TFRecordDataset([os.path.join(dirpath, 'test.tfrecords')]).map(extract_fn)
    
    #Shuffle train to determine labels
    ds = dataset_train.shuffle(buffer_size= NUM_EXAMPLES_TRAIN,seed = label_seed,reshuffle_each_iteration=False)
    dataset_train_l = ds.take(FLAGS.num_labeled_examples)
    dataset_train_u = ds.skip(FLAGS.num_labeled_examples)
    
    dataset_val = dataset_train.shuffle(buffer_size=NUM_EXAMPLES_TEST,seed = val_seed,reshuffle_each_iteration=False)
    dataset_val_l = dataset_train.take(FLAGS.num_valid_examples)
    dataset_val_u = dataset_train.skip(FLAGS.num_valid_examples)
    
    
    return dict(\
        train = tf.data.TFRecordDataset([os.path.join(dirpath, 'train.tfrecords')]).map(extract_fn),
        train_l = dataset_train_l,
        train_u = dataset_train_u,
        test = dataset_test,
        val = dataset_val,
        val_l = dataset_val_l,
        val_u = dataset_val_u)

def runSetOfExperiments(argv):
    prepare_dataset()


if __name__ == "__main__":
    tf.app.run()
