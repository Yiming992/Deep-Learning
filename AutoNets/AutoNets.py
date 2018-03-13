"""
Codes used to build and train autoencoder in the notebook
"""

import tensorflow as tf
import numpy as np

class Layers:
    """
    General class for both encoder and decoder, to store their common attributes.

    Attributes:
        num_layer: int, number of densely connected Layers
        units: tuple or list, number of neurons for each layer
        ksize: int, width and height of convolutional filter
        stride: int, number of step taken each time a filter moves
        activation: activation function for the layers
        initializer: tensorflow variable initializer to initialize weights for each layer
        mode: str, type of autoencoder, can be 'Dense', 'Variation' or 'Convolution'
    """
    def __init__(self,num_layer,units,ksize=None,stride=None,mode='Dense',activation=None,initializer=None):

        self.num_layer=num_layer
        self.units=units
        self.ksize=ksize
        self.stride=stride
        self.activation=activation
        self.initializer=initializer
        self.mode=mode


class Encoder(Layers):
    """
    Child class of class Layers. Responsible for building the encoder layers of the densesly connected autoencoder i.e. from input layer(not included) until middle encoding layer(included)
    No special attributes at this moment
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)



    def _variation_build(self,Input):
        """
        Internal method to build encoder part of the variational
        autoencoder
        """
        for i in range(self.num_layer):
            if i==self.num_layer-1:
                M=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=None,name='M')
                S=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=None,name='S')
            else:
                Input=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=self.activation)
        mu=tf.random_normal(tf.shape(M))
        Input=M+S*mu
        return Input,M,S
        

    def build(self,Input):
        """
        Method to build the encoder part of the tensorflow computational graph

        Argument:
            Input: a placeholder tnesor
        Output:
            codings: middle encoding layer output tensor
        """
        if self.mode=='Variation':
            Input,M,S =self._variation_build(Input)
            return Input,M,S
        for i in range(self.num_layer):
            if self.mode=='Convolution':
                if i==self.num_layer-1:
                    Input=tf.layers.conv2d(Input,self.units[i],kernel_size=self.ksize,strides=self.stride,
                          kernel_initializer=self.initializer,activation=self.activation,padding='same',name='Encodings')
                else:
                    Input=tf.layers.conv2d(Input,self.units[i],kernel_size=self.ksize,strides=self.stride,
                            kernel_initializer=self.initializer,activation=self.activation,padding='same')

            elif self.mode=='Dense':

                if i==self.num_layer-1:
                    Input=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=self.activation,name='Encodings')
                else:
                    Input=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=self.activation)
        codings=Input
        return codings



class Decoder(Layers):
    """
    Child class of class Layers. Responsible for building the decoder layers of the densely connected autoencoder i.e. from middle encoding layer(not included) to final output layer(included)
    """

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)



    def build(self,Input):
        """
        Method to build decoder part of the tensorflow computational graph

        Argument:
            Input: output tensor from final encoder layer

        Output:
            reconstruction: tensor representing the recreated input of the encoder layer
        """
        for i in range(self.num_layer):
            if self.mode=='Dense' or self.mode=='Variation':
                if i==self.num_layer-1:
                    Input=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=self.activation,name='Reconstructions')
                else:
                    Input=tf.layers.dense(Input,self.units[i],kernel_initializer=self.initializer,activation=self.activation)
                
            elif self.mode=='Convolution':
                if i==self.num_layer-1:
                    Input=tf.layers.conv2d_transpose(Input,self.units[i],kernel_size=self.ksize,strides=self.stride,
                        activation=self.activation,kernel_initializer=self.initializer,padding='same',name='Reconstructions')
                else:
                    Input=tf.layers.conv2d_transpose(Input,self.units[i],kernel_size=self.ksize,strides=self.stride,
                            activation=self.activation,kernel_initializer=self.initializer,padding='same')

        reconstruction=Input
        return reconstruction



class Autoencoder:
    """
    Class to build and train an densely connected autoencoder

    Attributes:
        inputs:int, Dimension of the input data.
        encoder_layer:int, number of layers in the encoder part of the network
        decoder_layer:int, number of layers in the decoder part of the network
        encoder_size:tuple or list, number of neurons for each layer of the encoder
        decoder_size:tuple or list, number of neurons for each layer of the decoder
        ksize:int,kernel size for the convolutional kernel
        stride:int, number of step each the filter moves
        activation:layer activation function used in the network. Default None will use linear activation
        initializer:tensorflow weights initializer. Default None will use glorot uniform
        sparsity_parameter:float or None(default), weight which determines the level sparsity penalty
        mode: str, type of autoencoder, can be 'Dense', 'Variation' or 'Convolution'
        sparse:Bool, weather to penalize the model with KL divergence
        save_path:str, file path where the trained model will be stored
        cross_entropy: bool, whether to use cross_entropy loss in Variational Autoencoder
    """

    def __init__(self,inputs,encoder_layer,decoder_layer,encoder_size,
        decoder_size,ksize=None,stride=None,activation=None,initializer=None, sparsity_parameter=None,
        mode='Dense',sparse=False,save_path='./saved_model/auto',cross_entropy=False):
        self.inputs=inputs
        self.encoder_layer=encoder_layer
        self.decoder_layer=decoder_layer
        self.encoder_size=encoder_size
        self.decoder_size=decoder_size
        self.ksize=ksize
        self.stride=stride
        self.activation=activation
        self.initializer=initializer
        self.sparsity=sparsity_parameter
        self.mode=mode
        self.sparse=sparse
        self.save_path=save_path
        self.cross_entropy=cross_entropy



    @staticmethod
    def KL_divergence(p,q):
        """
        Method to implement KL divergence penalty for sparse autoencoder
        """
        return p*tf.log(p/q)+(1-p)*tf.log((1-p)/(1-q))



    @staticmethod
    def get_batch(feature_tensor,batch_size):
        """
        Method to generate data batch by batch
        """

        num_batches=np.ceil(feature_tensor.shape[0]/batch_size)
        for i in range(int(num_batches)):
            if i!=num_batches-1:
                batch_array=feature_tensor[i*batch_size:(i+1)*batch_size,:]
            else:
                batch_array=feature_tensor[i*batch_size:(i+1)*batch_size,:]
            yield batch_array

    @staticmethod
    def scale_data(Input,method='Min-Max'):
        """
        Method to scale data
        """
        if method=='Min-Max':
            Max=Input.max(axis=0)
            Min=Input.min(axis=0)
            Output=(Input-Min)/(Max-Min)
        elif method=='Normal':
            mean=Input.mean(axis=0)
            std=Input.std(axis=0)
            Output=(Input-mean)/std
        return Output



    def build(self,lr=1e-3):
        """
        Method to build the computational graph for the densely connected autoencoder

        Argument:
            lr: learning rate for the optimizer

        Output:
            a python dictionary which contains various important parts of the network
        """
        log_file='./tf-logs/'
        Graph=tf.Graph()
        encoder=Encoder(self.encoder_layer,self.encoder_size,ksize=self.ksize,stride=self.stride,mode=self.mode,activation=self.activation,initializer=self.initializer)
        decoder=Decoder(self.decoder_layer,self.decoder_size,ksize=self.ksize,stride=self.stride,mode=self.mode,activation=self.activation,initializer=self.initializer)
        with Graph.as_default():
            if self.mode=='Convolution':
                Input=tf.placeholder(tf.float32,[None,self.inputs[0],self.inputs[1],self.inputs[2]])
            else:
                Input=tf.placeholder(tf.float32,[None,self.inputs])
            if self.mode=='Variation':
                codings,mean,std=encoder.build(Input)
            else:
                codings=encoder.build(Input)
            reconstruction=decoder.build(codings)
            with tf.name_scope('train'):
                optimizer=tf.train.AdamOptimizer(learning_rate=lr)
                if self.mode=='Dense':
                    if self.sparse:
                        Mean= tf.reduce_mean(codings,axis=0)
                        loss=tf.reduce_mean(tf.square(reconstruction-Input))+tf.reduce_sum(self.KL_divergence(self.sparsity,Mean))
                    else:
                        loss=tf.reduce_mean(tf.square(reconstruction-Input))
                elif self.mode=='Variation':
                
                    KL=-0.5*tf.reduce_sum(1+tf.log(std**2+1e-7)-mean**2-std**2,1)
                    if self.cross_entropy==True:
                        cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=Input,logits=reconstruction)
                        loss=tf.reduce_mean(KL+cross_entropy)

                    else:
                        mse=tf.reduce_sum(tf.square(reconstruction-Input),1)
                        loss=tf.reduce_mean(mse+KL)
                elif self.mode=='Convolution':
                    loss=tf.reduce_mean(tf.square(reconstruction-Input))
            train_op=optimizer.minimize(loss)
            tf.summary.FileWriter(log_file,Graph)
            init=tf.global_variables_initializer()
        return {'model':Graph,'input':Input,'coding':codings,'loss':loss,'training':train_op,'init':init}


    def train(self,input_tensor,num_epochs,batch_size,model,return_pc=False,show_train=True):
        """
        Method to train and save the network

        Arguments:
            input_tensor:ndarray, input data
            num_epochs:int, number of epochs
            batch_size:int, number of samples within each batch
            model:build method output
            return_pc:bool, whether to return middle layer encodings of the input data. default None
            show_train: bool, whether to print out loss during training 
        """

        with tf.Session(graph=model['model']) as sess:
            sess.run(model['init'])
            for i in range(num_epochs):
                for batch in self.get_batch(input_tensor,batch_size):
                    sess.run(model['training'],feed_dict={model['input']:batch})
                if show_train==True:
                    print('Number of epchs: {}/{} Loss: {}'.format(i+1,num_epochs,sess.run(model['loss'],feed_dict={model['input']:input_tensor[:500,:]})))
            if return_pc==True:
                PCs=[]
                for batch in self.get_batch(input_tensor,batch_size):
                    PCs.append(sess.run(model['coding'],feed_dict={model['input']:batch}))
                return np.concatenate(PCs)
            else:
                Saver=tf.train.Saver()
                Saver.save(sess,self.save_path)
                print('Model trained and saved')


class Transformer:
    """
    Class to use the trained autoencoder to reduce the dimension of the input data

    Attributes:
        data:ndarray, target of the dimensionality reduction
        save_path:str, path where saved model resides
        graph:tensorflow graph object, will be infered from the save_path
        saver:tensorflow Saver object, will be automatically infered
    """
    def __init__(self,data,save_path='./saved_model/auto',mode='Dense'):
        self.save_path=save_path
        self.data=data
        self.mode=mode
        loaded_graph=tf.Graph()
        with loaded_graph.as_default():
            saver=tf.train.import_meta_graph('{}.meta'.format(self.save_path))
        self.graph=loaded_graph
        self.saver=saver

    def transform(self):
        """
        Method to obtain the encodings of the input data
        with trained autoencoder

        """
        with tf.Session(graph=self.graph) as sess:
            self. saver.restore(sess,tf.train.latest_checkpoint(self.save_path[:-4]))
            if self.mode=='Variation':
                encodings=tf.get_default_graph().get_tensor_by_name('add:0')
            elif self.mode=='Convolution':
                encodings=tf.get_default_graph().get_tensor_by_name('Encodings/Relu:0')
            else:
                encodings=tf.get_default_graph().get_tensor_by_name('Encodings/BiasAdd:0')
            if self.mode=='Convolution':
                reconstruction=tf.get_default_graph().get_tensor_by_name('Reconstructions/Relu:0')
            else:
                reconstruction=tf.get_default_graph().get_tensor_by_name('Reconstrcutions/BiasAdd:0')
            Input=tf.get_default_graph().get_tensor_by_name('Placeholder:0')

            test_encodings,reconstruction=sess.run([encodings,reconstruction],feed_dict={Input:self.data})
            return test_encodings,reconstruction
