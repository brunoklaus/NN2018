import tensorflow as tf
from dataset_utils import extract_fn, count_ds_size
import numpy as np
import gssl_affmat as affmat
import plotly.offline as pyoff
import plotly.io as pio
import plotly.graph_objs as go
import os

def plot_knn_sample(affmat_path,ds_path,output_dir,num_points = 5,k= None):
    K, _ = load_whole_affmat(affmat_path)
    if not k is None:
        K = K[:,0:k]

    print("Affmat shape:{}".format(K.shape))
    
    ds_dict = load_whole_dataset(ds_path, get_images=True, get_zcas=True, get_labels=False, get_emb=False)
    all_images = ds_dict.pop("image")
    all_zca = ds_dict.pop("zca")
    

    
    
    def map_to_image(P):
        j = tf.cast(tf.gather(P,1),tf.int64)
        img = tf.gather(all_images,j)
        return 255.0*(img + 0.5)
    
    def map_to_zca(P):
        j = tf.cast(tf.gather(P,1),tf.int64)
        img = tf.gather(all_zca,j)
        M = tf.reduce_max(img)
        m = tf.reduce_min(img)
        return 255.0*(img - m)/(M-m)
    def map_to_j(P):
        j = tf.cast(tf.gather(P,1),tf.int64)
        return(tf.cast(j,tf.float32))
    #Plot knn for 1st point
    sample = list()
    tsr = tf.placeholder(shape=K[0,:,:].shape,dtype = tf.float32)

    sample_img = tf.map_fn(lambda x:map_to_image(x),tsr)
    sample_img = tf.cast(sample_img,tf.uint8)
    sample_img = tf.reshape(sample_img,[-1,32,3])
    
    sample_zca = tf.map_fn(lambda x:map_to_zca(x),tsr)
    sample_zca = tf.cast(sample_zca,tf.uint8)
    sample_zca = tf.reshape(sample_zca,[-1,32,3])
    with tf.Session() as sess:
        for i in range(num_points):
            s_img,s_zca = sess.run([sample_img,sample_zca],feed_dict={tsr:K[i,:,:]})
            sample.extend([s_img,s_zca])
            
        sample = tf.concat(sample,axis=1)
        encoded_sample = tf.image.encode_png(sample)
        sess.run([tf.write_file(os.path.join(output_dir, "knn.png"),encoded_sample)  ])
        print("saved knn sample img")
        

def get_knn_acc(affmat_path,ds_path,k= None,ds_size = None):
    k = k + 1
    K, _ = load_whole_affmat(affmat_path,ds_size = ds_size)
    if not k is None:
        K = K[:,0:k,:]
    
    #Remove neighborhood to self 
    K = K[:,1:,:]   
    print("Affmat shape:{}".format(K.shape))
    
    def map_to_label(P):
        j = tf.cast(tf.gather(P,1),tf.int64)
        return tf.cast(tf.gather(labels,j),tf.int64)

    ds_dict = load_whole_dataset(ds_path, get_images=False, get_zcas=False, get_labels=True,
                                  get_emb=False,one_hot_label=False,ds_size = ds_size)
    labels = ds_dict.pop("label")
    
    L = np.zeros((K.shape[0],K.shape[1]),dtype=np.int64)
    for i in np.arange(K.shape[0]):
        for j in np.arange(K.shape[1]):
            L[i,j] = labels[K[i,j,1]]
    
    correct = 0
    for i in np.arange(K.shape[0]):
        if labels[i] == np.argmax(np.bincount(L[i,])):
            correct += 1
        
    return correct / K.shape[0]

    
def load_whole_dataset(dirpath, get_images=True,get_zcas=True,get_labels=True,get_emb = True,batch_size=1000,
                 ds_size=None,one_hot_label = True):


    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    dataset = tf.data.TFRecordDataset([dirpath]).map(extract_fn)    
    bs = batch_size   
    if ds_size is None:
        ds_size = count_ds_size(dataset,bs) 
    dataset = dataset.batch(bs)
    

    
    #Read dataset
    with tf.Session(config=tfconfig) as sess:
        it = dataset.make_one_shot_iterator()
        nxt_op = it.get_next()
        for i in range(ds_size//bs):
            nxt = sess.run(nxt_op)
            if not (nxt["id"][0] == i * bs):
                raise ValueError("Dataset was not saved in order")                
                
            if i == 0:
                all_emb = np.zeros([ds_size,nxt["emb"].shape[1]]) if get_emb else None
                all_images = np.zeros([ds_size,32,32,3],dtype=np.float32) if get_images else None
                all_zca = np.zeros([ds_size,32,32,3],dtype=np.float32) if get_zcas else None
                if one_hot_label:
                    all_labels = np.zeros([ds_size,nxt["label"].shape[1]],dtype=np.int64) if get_labels else None
                else:
                    all_labels = np.zeros([ds_size],dtype=np.int64) if get_labels else None
                
            if get_images:
                all_images[i*bs:(i+1)*bs,:] = nxt["image"]
            if get_zcas:
                all_zca[i*bs:(i+1)*bs,:] = nxt["zca"]
            if get_emb:
                all_emb[i*bs:(i+1)*bs,:] = nxt["emb"]
            if get_labels:
                if one_hot_label:
                    all_labels[i*bs:(i+1)*bs,:] = nxt["label"]
                else:
                    all_labels[i*bs:(i+1)*bs] = np.argmax(nxt["label"],axis=1)
                
           
        return {"image":all_images,"zca":all_zca,"label":all_labels,
                "emb":all_emb}
            
def load_affmat(input_path):
    ds = tf.data.TFRecordDataset([input_path]).map(extract_affmat_fn)
    return ds
def load_whole_affmat(input_path, batch_size = 10000,ds_size = None):
    bs = batch_size
    ds = tf.data.TFRecordDataset([input_path]).map(extract_affmat_fn)
    if ds_size is None:
        ds_size = count_ds_size(ds,bs) 
    ds = ds.batch(bs)    
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5
    
    #Read affmat
    with tf.Session(config=tfconfig) as sess:
        it = ds.make_one_shot_iterator()
        nxt_op = it.get_next()
        for i in range(ds_size//bs):
            nxt = sess.run(nxt_op)
            if i == 0:
                K = np.zeros((ds_size,nxt["pair"].shape[1]),dtype = np.int64)
                DIST = np.zeros((ds_size,nxt["dist"].shape[1]))
            K[i*bs:(i+1)*bs,:] = nxt["pair"]
            DIST[i*bs:(i+1)*bs,:] = nxt["dist"]
    K = np.reshape(K,(ds_size,K.shape[1]//2,2))    
    return K, DIST 

def extract_affmat_fn(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'pair': tf.FixedLenSequenceFeature([],allow_missing=True,dtype=tf.int64),
            'dist': tf.FixedLenSequenceFeature([],allow_missing=True,dtype=tf.float32)
        })
    
    return {"pair":features["pair"],"dist":features["pair"]}
 


def create_sparsemat(k,dirpath,output_dir=None, use_emb=True,use_zca=False):
    
        ds_dict = load_whole_dataset(dirpath, get_images=True, get_zcas=True, get_labels=False, get_emb=True)
        all_emb = ds_dict.pop("emb")
        all_images = ds_dict.pop("image")
        all_zca = ds_dict.pop("zca")
        
        
        if use_emb:
            K, Ds = affmat.knnMask_sparse(all_emb, k)
        else:
            if use_zca:
                K, Ds = affmat.knnMask_sparse(np.reshape(all_zca,[all_emb.shape[0],-1]), k)
            else:
                K, Ds = affmat.knnMask_sparse(np.reshape(all_images,[all_emb.shape[0],-1]), k)
        print("Teste")
       
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        convert_sparse_Affmat(K, Ds, os.path.join(output_dir, "Affmat.tfrecords"))
                        
    
def convert_sparse_Affmat(K,Ds,filepath):
    if not  K.shape[0] == Ds.shape[0]:
        raise ValueError("K, Ds shapes not compatible")
    
    writer = tf.python_io.TFRecordWriter(filepath)
    for index in range(K.shape[0]):
        P = np.reshape(K[index,:,:],-1).astype(np.int64).tolist()
        dist = Ds[index,:].tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
            'pair':  tf.train.Feature(int64_list=tf.train.Int64List(value=P)),
            'dist': tf.train.Feature(float_list=tf.train.FloatList(value=dist))}))
        writer.write(example.SerializeToString())
    writer.close()

def runSetOfExperiments(_):
    #create_sparsemat(k=100,dirpath="../dataset/cifar10/AE_LR=0.00001.tfrecords",\
    #                 output_dir="../affmat/AE_LR=0.00001/k=100")
    #create_sparsemat(k=100,dirpath="../dataset/cifar10/AE_ZCA_LR=0.001.tfrecords",\
    #                 output_dir="../affmat/AE_ZCA_LR=0.001/k=100",use_zca=True,use_emb=False)
    #create_sparsemat(k=100,dirpath="../dataset/cifar10/AE_LR=0.001.tfrecords",\
    #                 output_dir="../affmat/AE_FULL/k=100",use_emb=False)
    #create_sparsemat(k=100,dirpath="../dataset/cifar10/AE_ZCA_LR=0.001.tfrecords",\
    #                 output_dir="../affmat/AE_ZCA_FULL/k=100",use_emb=False)
    create_sparsemat(k=100,dirpath="../dataset/svhn/AE_embedding.tfrecords",\
                     output_dir="../affmat/svhn/AE_embedding/k=100",use_emb=True)
    #create_sparsemat(k=100,dirpath="../dataset/svhn/train.tfrecords",\
    #                 output_dir="../affmat/svhn/AE_FULL/k=100",use_emb=False)
    
    plot_knn_sample(affmat_path="../affmat/svhn/AE_embedding/k=100/Affmat.tfrecords",
                     ds_path="../dataset/svhn/train.tfrecords",
                     output_dir="../affmat/svhn/AE_FULL/k=100",
                      num_points=10, k=20)
    
    #return True
    
    
    ''' plot_knn_sample(k=20,num_points=10,
                    affmat_path="../affmat/AE_ZCA_LR=0.00001/k=100/Affmat.tfrecords",
                     ds_path="../dataset/cifar10/train.tfrecords",
                    output_dir="../affmat/AE_LR=0.00001/k=100/")
    '''
    

    
    all_affmat_strs =\
    ["../affmat/cifar10/AE_ZCA_LR=0.001/k=100/Affmat.tfrecords",
     "../affmat/cifar10/AE_LR=0.001/k=100/Affmat.tfrecords",
     "../affmat/cifar10/AE_FULL/k=100/Affmat.tfrecords",
     "../affmat/AE_ZCA_LR=0.001/k=100/Affmat.tfrecords"]
    names = ["ZCA + AE","NO_ZCA + AE","NO_ZCA","ZCA"]
    
    all_affmat_strs =\
    ["../affmat/svhn/AE_embedding/k=100/Affmat.tfrecords",
     "../affmat/svhn/AE_FULL/k=100/Affmat.tfrecords"]
    names = ["SVHN_AE","SVHN"]
    
   
    data = []
    for j in range(len(all_affmat_strs)):
        ks = [1,3,5,7] + np.arange(10,99,10,dtype=np.int64).tolist()
        accs = np.zeros(len(ks))
        for i in range(len(ks)):
            
            print("Affmat:{};K={}".format(all_affmat_strs[j],ks[i]))
            accs[i] = get_knn_acc(affmat_path=all_affmat_strs[j], 
                    ds_path="../dataset/svhn/train.tfrecords",
                     k=ks[i],ds_size=73257)
        data.append(
            go.Scatter(
            x = ks,
            y = accs,
            mode = 'lines',
            name = names[j]
            )
            )
        
    layout = go.Layout(
                 title="knn accuracies",
                 xaxis=dict(
                    autorange=True
                ),
                yaxis=dict(
                    autorange=True
                    )
                )    
    pyoff.offline.plot(data)
    pio.write_image(go.Figure(data,layout),"../knn_accuracies.png")

if __name__ == '__main__':
    tf.app.run()