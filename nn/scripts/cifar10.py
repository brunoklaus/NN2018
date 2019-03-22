# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from scipy import linalg
import glob
import pickle

from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib

import tensorflow as tf

from dataset_utils import *

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../dataset/cifar10',
                           'where to store the dataset')
tf.app.flags.DEFINE_integer('num_labeled_examples', 4000, "The number of labeled examples")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "The number of validation examples")



# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_TRAIN = 50000
NUM_EXAMPLES_TEST = 10000


def load_cifar10():
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename, float(count * block_size) /
                              float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    # Training set
    print("Loading training data...")
    train_images = np.zeros((NUM_EXAMPLES_TRAIN, 3 * 32 * 32), dtype=np.float32)
    train_labels = []
    for i, data_fn in enumerate(
            sorted(glob.glob(FLAGS.data_dir + '/cifar-10-batches-py/data_batch*'))):
        batch = unpickle(data_fn)
        train_images[i * 10000:(i + 1) * 10000] = batch['data']
        train_labels.extend(batch['labels'])
    train_images = (train_images - 127.5) / 255.
    train_labels = np.asarray(train_labels, dtype=np.int64)


    print("Loading test data...")
    test = unpickle(FLAGS.data_dir + '/cifar-10-batches-py/test_batch')
    test_images = test['data'].astype(np.float32)
    test_images = (test_images - 127.5) / 255.
    test_labels = np.asarray(test['labels'], dtype=np.int64)
    

    
    print("Apply ZCA whitening")
    components, mean, train_zca = ZCA(train_images)
    print("var:{}".format(np.var(train_images)))
    print("mean:{}".format(np.mean(train_images)))
    np.save('{}/components'.format(FLAGS.data_dir), components)
    np.save('{}/mean'.format(FLAGS.data_dir), mean)
    test_zca = np.dot(test_images - mean, components.T)

    train_images = train_images.reshape(
        (NUM_EXAMPLES_TRAIN, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TRAIN, -1))
    test_images = test_images.reshape(
        (NUM_EXAMPLES_TEST, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TEST, -1))

    train_zca = train_zca.reshape(
        (NUM_EXAMPLES_TRAIN, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TRAIN, -1))
    test_zca = test_zca.reshape(
        (NUM_EXAMPLES_TEST, 3, 32, 32)).transpose((0, 2, 3, 1)).reshape((NUM_EXAMPLES_TEST, -1))
    

    return (train_images, train_zca,train_labels), (test_images,test_zca, test_labels)


def prepare_dataset():
    (train_images, train_zca, train_labels), (test_images, test_zca, test_labels) = load_cifar10()
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
