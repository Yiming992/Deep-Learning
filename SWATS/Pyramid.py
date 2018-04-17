import numpy as np
import tensorflow as tf
import argparse
import pickle
from progress import progress
from SWATS import SWATS

class Pyramid_110():

    """
    This class will implement the 110-layer deep pyramidal residual network as described in paper "Deep Pyramidal Residual Networks", which can be found at https://arxiv.org/abs/1610.02915
    """

    def __init__(self,alpha):
        """
        alpha:widening factor of the network
        """
        self.alpha=alpha

    def _Input(self):
        """
        Create network inputs
        """
        return tf.placeholder(tf.float32,[None,None,None,3]),tf.placeholder(tf.float32,[None,10]),tf.placeholder(tf.bool)

    def _first_conv(self,inputs):
        """
        First convolutional layer
        """
        return tf.layers.conv2d(inputs,16,3,strides=1,activation=tf.nn.relu)

    def _basic_block(self,inputs,k,is_train,downsample=False):
        """
        Basic residual block as described in the paper
        """
        shortcut=inputs
        channels=tf.shape(inputs)[-1]
        out_depth=np.ceil(16+self.alpha*(k-1)/54)
        inputs=tf.layers.batch_normalization(inputs,training=is_train)
        if downsample:
            # if downsample is required a stride 2 convolution is used
            shortcut=tf.layers.conv2d(shortcut,out_depth,1,strides=2,padding='same')
            inputs=tf.layers.conv2d(inputs,out_depth,3,strides=2,padding='same')
        else:
            inputs=tf.layers.conv2d(inputs,out_depth,3,strides=1,padding='same')
            shortcut=tf.pad(shortcut,[[0,0],[0,0],[0,0],[0,out_depth-channels]])
        inputs=tf.layers.batch_normalization(inputs,training=is_train)
        inputs=tf.nn.relu(inputs)
        inputs=tf.layers.conv2d(inputs,out_depth,3,strides=1,padding='same')
        inputs=tf.layers.batch_normalization(inputs,training=is_train)
        inputs+=shortcut
        return inputs

    def _output(self,inputs,label):
        """
        Method to create outputs of network

        """
        avg_pool=tf.layers.conv2d(inputs,16+self.alpha,8,strides=8,padding='same')
        avg_pool=tf.squeeze(avg_pool,axis=[1,2])
        results=tf.layers.dense(avg_pool,10)
        probs=tf.nn.softmax(results)
        loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=results,labels=label)
        bools=tf.equal(tf.argmax(probs,axis=1),tf.argmax(label,axis=1))
        correct_count=tf.reduce_sum(tf.cast(bools,tf.float32))
        return probs,loss,correct_count

    def build(self,optimizer='Adam'):
        """
        Method to construct the computational graph
        """
        Graph=tf.Graph()
        num_units=54
        with Graph.as_default():
            Input,label,is_train=self._Input()
            inputs=self._first_conv(Input)
            for i in range(num_units):
                if i ==19 or i==37:
                    with tf.variable_scope('Downsample_block_{}'.format(np.ceil(i/19))):
                        inputs=self._basic_block(inputs,i,is_train,downsample=True)
                else:
                    with tf.variable_scope('Residual_block_{}'.format(i+1)):
                        inputs=self._basic_block(inputs,i,is_train,downsample=False)
            probs,loss,correct_count=self._output(inputs,label)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                if optimizer=='Adam':
                    train_op=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
                elif optimizer=='Swats':
                    train_op=SWATS().minimize(loss)
            init=tf.global_variables_initializer()
        return Graph,Input,label,is_train,correct_count,train_op,init


if __name__=='__main__':
    # Load in cifar-10 test and train data
    train_image=np.load('./cifar_train.pk')
    train_label=np.load('./Labels.pk')
    test_image=np.load('./test_data.pk')
    test_label=np.load('./test_label.pk')

    # Generators to generate batchs of data
    def get_train_batch(images,labels,batch_size,seed):
        np.random.seed(seed)
        images=np.random.permutation(images)
        np.random.seed(seed)
        labels=np.random.permutation(labels)
        num_batch=images.shape[0]//batch_size
        for i in range(num_batch):
            image=images[i*batch_size:(i+1)*batch_size]
            label=labels[i*batch_size:(i+1)*batch_size]
            image=image.astype(np.float32)
            label=label.astype(np.float32)
            yield image,label

    def get_test_batch(images,labels):
        num_batch=images.shape[0]//100
        for i in range(num_batch):
            image=images[i*100:(i+1)*100]
            label=labels[i*100:(i+1)*100]
            image=image.astype(np.float32)
            label=label.astype(np.float32)
            yield image,label


    # Create necessary command line arguments
    ap=argparse.ArgumentParser()

    ap.add_argument('-b','--batch_size',required=True)
    ap.add_argument('-e','--epoch',required=True)
    ap.add_argument('-o','--optimizer',default='Adam',required=False)

    args=vars(ap.parse_args())

    batch_size=int(args['batch_size'])
    epochs=int(args['epoch'])
    optimizer=args['optimizer']

    # Initialize the network
    pyramid=Pyramid_110(48)
    graph,inputs,label,is_train,correct_count,train_op,init=pyramid.build(optimizer=optimizer)

    with tf.Session(graph=graph) as sess:
        sess.run(init)
        AC=[]
        for i in range(epochs):
            correct=0
            for image,labels in get_train_batch(train_image,train_label,batch_size,i):
                sess.run(train_op,feed_dict={inputs:image,label:labels,is_train:True})
            for image,labels in get_test_batch(test_image,test_label):
                count=sess.run(correct_count,feed_dict={inputs:image,label:labels,is_train:False})
                correct+=count
            accuracy=correct/10000
            AC.append(1-accuracy)
            if i!=epochs-1:
                progress(i+1,int(args['epoch']),status='Training')
            else:
                progress(i+1,int(args['epoch']),status='Finished')
        # dump the test accuracy to disk
        pickle.dump(np.array(AC),open('accuracy.pk','wb'))
