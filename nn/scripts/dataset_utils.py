import tensorflow as tf
import os, sys, pickle
import numpy as np
from scipy import linalg



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('aug_trans', True, "")
tf.flags.DEFINE_bool('aug_flip', True, "")

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


def ZCA(data, reg=1e-6):
    mean = np.mean(data, axis=0)
    mdata = data - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S) + reg)), U.T)
    whiten = np.dot(data - mean, components.T)
    return components, mean, whiten


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_images_and_labels(images, labels,zcas, filepath):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                         (images.shape[0], num_examples))
    if zcas.shape[0] != num_examples:
        raise ValueError("zca  size %d does not match label size %d." %
                         (zcas.shape[0], num_examples))
    if not np.all(zcas.shape == images.shape):
        raise ValueError("zca  shape {} differs from image shape{}.".format(zcas.shape,images.shape))

    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        zca = zcas[index].tolist()
        
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        zca_feature = tf.train.Feature(float_list=tf.train.FloatList(value=zca))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature,
            'zca': zca_feature,
            'id':_int64_feature(index)
            }))
        writer.write(example.SerializeToString())
    writer.close()
    
    

def convert_images_and_labels_and_emb(images, labels,zcas,embs,filepath):
    num_examples = labels.shape[0]
    print("{} examples".format(num_examples))
    print('Writing', filepath)
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(num_examples):
        image = images[index].tolist()
        zca = zcas[index].tolist()
        emb = embs[index].tolist()
        emb_feature = tf.train.Feature(float_list=tf.train.FloatList(value=emb))
        image_feature = tf.train.Feature(float_list=tf.train.FloatList(value=image))
        zca_feature = tf.train.Feature(float_list=tf.train.FloatList(value=zca))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(32),
            'width': _int64_feature(32),
            'depth': _int64_feature(3),
            'label': _int64_feature(int(labels[index])),
            'image': image_feature,
            'emb': emb_feature,
            'id':_int64_feature(index),
            'zca': zca_feature}))
        writer.write(example.SerializeToString())
    writer.close()


def read(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([3072], tf.float32),
            'zca': tf.FixedLenFeature([3072], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
            
        })

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    image = tf.reshape(image, [32, 32, 3])
    zca = features['zca']
    zca = tf.reshape(zca,[32,32,3])
    
    label = tf.one_hot(tf.cast(features['label'], tf.int32), 10)
    return image, label, zca

def extract_fn(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([3072], tf.float32),
            'zca': tf.FixedLenFeature([3072], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),
            'id':tf.FixedLenFeature([], tf.int64),
            'emb': tf.FixedLenSequenceFeature([],allow_missing=True,dtype=tf.float32)
        })

    
    # Convert label from a scalar uint8 tensor to an int32 scalar.
    image = features['image']
    image = tf.reshape(image, [32, 32, 3])
    zca = features['zca']
    zca = tf.reshape(zca,[32,32,3])
    label = tf.reshape(features['label'],())
    label = tf.one_hot(tf.cast(label, tf.int32), 10)
    return {"image":image, "label":label,"zca":zca,"emb":features["emb"],"id":features["id"]}



def generate_batch(
        example,
        min_queue_examples,
        batch_size, shuffle):
    """
    Arg:
        list of tensors.
    """
    num_preprocess_threads = 1

    if shuffle:
        ret = tf.train.shuffle_batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        ret = tf.train.batch(
            example,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            allow_smaller_final_batch=True,
            capacity=min_queue_examples + 3 * batch_size)
        

    return ret


def transform(image):
    image = tf.reshape(image, [32, 32, 3])
    if FLAGS.aug_trans or FLAGS.aug_flip:
        print("augmentation")
        if FLAGS.aug_trans:
            image = tf.pad(image, [[2, 2], [2, 2], [0, 0]])
            image = tf.random_crop(image, [32, 32, 3])
        if FLAGS.aug_flip:
            image = tf.image.random_flip_left_right(image)
    return image


def generate_filename_queue(filenames, data_dir, num_epochs=None):
    print("filenames in queue:", filenames)
    for i in range(len(filenames)):
        filenames[i] = os.path.join(data_dir, filenames[i])
    return tf.train.string_input_producer(filenames, num_epochs=num_epochs)

    

def count_ds_size(dataset,batch_size= 1000):
    NUM_EXAMPLES = 0
    #Read dataset
    print("Counting dataset size...")
    with tf.Session() as sess:
        it = dataset.batch(batch_size).make_one_shot_iterator()
        nxt_op = it.get_next()        
        while True:
            try:
                nxt = sess.run(nxt_op)
                NUM_EXAMPLES += nxt[list(nxt.keys())[0]].shape[0]
            except tf.errors.OutOfRangeError:
                break
    
    print("Done!")
    return NUM_EXAMPLES



