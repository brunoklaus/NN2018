import tensorflow as tf
import numpy as np
import layers as L
from dataset_utils import transform

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('keep_prob_hidden', 0.5, "dropout rate")
tf.flags.DEFINE_float('lrelu_a', 0.1, "lrelu slope")
tf.flags.DEFINE_boolean('top_bn', False, "")




def logit_small(x, num_classes, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):

    if is_training:
        scope = tf.name_scope("Training")
        
    else:
        scope = tf.name_scope("Testing")

    with scope:
        h = x
    
        rng = np.random.RandomState(seed)
    
        h = L.fc(h, dim_in=x.shape[1], dim_out = 64, seed=rng.randint(123456), name="fc1")
        h = L.lrelu(L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, name='fc1_normalized'), FLAGS.lrelu_a)
        h = L.fc(h, dim_in=64, dim_out = 64, seed=rng.randint(123456), name="fc2")
        h = L.lrelu(L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, name='fc2_normalized'), FLAGS.lrelu_a)
        h = L.fc(h, dim_in=64, dim_out = num_classes, seed=rng.randint(123456), name="fc3")
        return h
        

    
def logit(x,num_classes=10, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):

    if is_training:
        scope = tf.name_scope("Training")
        
    else:
        scope = tf.name_scope("Testing")

    with scope:
        h = x
    
        rng = np.random.RandomState(seed)
    
        h = L.conv(h, ksize=3, stride=1, f_in=3, f_out=128, seed=rng.randint(123456), name='c1')
        h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b1'), FLAGS.lrelu_a)
        h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c2')
        h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b2'), FLAGS.lrelu_a)
        h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=128, seed=rng.randint(123456), name='c3')
        h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b3'), FLAGS.lrelu_a)
    
        h = L.max_pool(h, ksize=2, stride=2)
        h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h
    
        h = L.conv(h, ksize=3, stride=1, f_in=128, f_out=256, seed=rng.randint(123456), name='c4')
        h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b4'), FLAGS.lrelu_a)
        h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=256, seed=rng.randint(123456), name='c5')
        h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b5'), FLAGS.lrelu_a)
        h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=256, seed=rng.randint(123456), name='c6')
        h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b6'), FLAGS.lrelu_a)
    
        h = L.max_pool(h, ksize=2, stride=2)
        h = tf.nn.dropout(h, keep_prob=FLAGS.keep_prob_hidden, seed=rng.randint(123456)) if stochastic else h
    
        h = L.conv(h, ksize=3, stride=1, f_in=256, f_out=512, seed=rng.randint(123456), padding="VALID", name='c7')
        h = L.lrelu(L.bn(h, 512, is_training=is_training, update_batch_stats=update_batch_stats, name='b7'), FLAGS.lrelu_a)
        h = L.conv(h, ksize=1, stride=1, f_in=512, f_out=256, seed=rng.randint(123456), name='c8')
        h = L.lrelu(L.bn(h, 256, is_training=is_training, update_batch_stats=update_batch_stats, name='b8'), FLAGS.lrelu_a)
        h = L.conv(h, ksize=1, stride=1, f_in=256, f_out=128, seed=rng.randint(123456), name='c9')
        h = L.lrelu(L.bn(h, 128, is_training=is_training, update_batch_stats=update_batch_stats, name='b9'), FLAGS.lrelu_a)
    
        h = tf.reduce_mean(h, reduction_indices=[1, 2])  # Global average pooling
        h = L.fc(h, 128, num_classes, seed=rng.randint(123456), name='fc')
    
        if FLAGS.top_bn:
            h = L.bn(h, num_classes, is_training=is_training,
                     update_batch_stats=update_batch_stats, name='bfc')
    
        return h
    
    



def autoencoder(x,zca, is_training=True, update_batch_stats=True, stochastic=True, seed=1234,use_zca=True):

    if is_training:
        scope = tf.name_scope("Training")
        
    else:
        scope = tf.name_scope("Testing")

    with scope:
        #Initial shape (-1, 32, 32, 3)
        x = x + 0.5 #Recover [0,1] range
        if use_zca:
            h = zca
        else:
            h = x
        print(h.shape)
        rng = np.random.RandomState(seed)
        
        #h = tf.map_fn(lambda x:transform(x),h)

        #(1) conv + relu + maxpool (-1, 16, 16, 64)
        h = L.conv(h, ksize=3, stride=1, f_in=3, f_out=64, seed=rng.randint(123456),padding="SAME", name='conv1')
        h = L.lrelu(L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, name='conv1_bn'), FLAGS.lrelu_a)
        h = L.max_pool(h, ksize=2, stride=2)

        #(2) conv + relu + maxpool (-1, 8, 8, 32)
        h = L.conv(h, ksize=3, stride=1, f_in=64, f_out=32, seed=rng.randint(123456),padding="SAME", name='conv2')
        h = L.lrelu(L.bn(h, 32, is_training=is_training, update_batch_stats=update_batch_stats, name='conv2_bn'), FLAGS.lrelu_a)
        h = L.max_pool(h, ksize=2, stride=2)
        

        #(3) conv + relu + maxpool (-1, 4, 4, 16)
        h = L.conv(h, ksize=3, stride=1, f_in=32, f_out=16, seed=rng.randint(123456),padding="SAME", name='conv3')
        h = L.lrelu(L.bn(h, 16, is_training=is_training, update_batch_stats=update_batch_stats, name='conv3_bn'), FLAGS.lrelu_a)
        h = L.max_pool(h, ksize=2, stride=2)
        
        
        encoded = h
        #(4) deconv + relu (-1, 8, 8, 16)
        h = L.deconv(encoded, ksize=5, stride=1, f_in=16, f_out=16,seed=rng.randint(123456), padding="SAME", name="deconv1")
        h = L.lrelu(L.bn(h, 16, is_training=is_training, update_batch_stats=update_batch_stats, name='deconv1_bn'), FLAGS.lrelu_a)
        

        #(5) deconv + relu (-1, 16, 16, 32)
        h = L.deconv(h, ksize=5, stride=1, f_in=16, f_out=32, padding="SAME",name= "deconv2")
        h = L.lrelu(L.bn(h, 32, is_training=is_training, update_batch_stats=update_batch_stats, name='deconv2_bn'), FLAGS.lrelu_a)
       
        

        #(5) deconv + relu (-1, 32, 32, 64)
        h = L.deconv(h, ksize=5, stride=1, f_in=32, f_out=64, padding="SAME",name= "deconv3")
        h = L.lrelu(L.bn(h, 64, is_training=is_training, update_batch_stats=update_batch_stats, name='deconv3_bn'), FLAGS.lrelu_a)
        


        #(7) conv + sigmoid (-1, 32, 32, 3)
        h = L.conv(h, ksize=3, stride=1, f_in=64, f_out=3, seed=rng.randint(123456),padding="SAME", name='convfinal')
        if use_zca:
            h = L.bn(h, 3, is_training=is_training, update_batch_stats=update_batch_stats, name='deconv4_bn')
        else:
            h = tf.sigmoid(h)

        num_samples=10
        sample_og_zca = tf.reshape(tf.slice(zca,[0,0,0,0],[num_samples,32,32,3]),(num_samples*32,32,3))
        sample_og_color = tf.reshape(tf.slice(x,[0,0,0,0],[num_samples,32,32,3]),(num_samples*32,32,3))
        sample_rec = tf.reshape(tf.slice(h,[0,0,0,0],[num_samples,32,32,3]),(num_samples*32,32,3))
        if use_zca:
            sample = tf.concat([sample_og_zca,sample_rec],axis=1)
            m = tf.reduce_min(sample)
            sample = (sample - m) / (tf.reduce_max(sample)-m)
        else:
            m = tf.reduce_min(sample_og_zca)
            sample_og_zca = (sample_og_zca - m) / (tf.reduce_max(sample_og_zca)-m)
            sample = tf.concat([sample_og_zca,sample_rec],axis=1)
        sample = tf.concat([sample_og_color,sample],axis=1)
        sample = tf.cast(255.0*sample,tf.uint8)

        
        
        
        if use_zca:
            loss = tf.reduce_mean(tf.losses.mean_squared_error(zca,h))
        else:
            loss = tf.reduce_mean(tf.losses.log_loss(x,h))
            
        
        return loss, encoded, sample    
