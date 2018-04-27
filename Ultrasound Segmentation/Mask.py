#height 420 width 580
import tensorflow as tf
import cv2
import glob
import os
import numpy as np

def sort_train_files(s):
    base=os.path.basename(s)
    num=int(base[:-5].replace('_',''))
    return num

def sort_mask_files(s):
    base=os.path.basename(s)
    num=int(base[:-9].replace('_',''))
    return num

train_images=glob.glob('train1/*.tif')
train_images.sort(key=sort_train_files)
train_mask=glob.glob('train_mask1/*tif')
train_mask.sort(key=sort_mask_files)

initializer=tf.variance_scaling_initializer()
def conv_2d(x,output_nums,conv_size,stride,activation='relu'):
    if activation=='relu':
        output=tf.layers.conv2d(x,output_nums,conv_size,strides=stride,padding='same',kernel_initializer=initializer)
        output=tf.nn.relu(output)
    elif activation=='sigmoid':
        output=tf.layers.conv2d(x,output_nums,conv_size,strides=stride,padding='same')
        output=tf.nn.sigmoid(output)
    else:
        output=tf.layers.conv2d(x,output_nums,conv_size,strides=stride,padding='same')
    return output

def max_pool2d(x,k_size,stride):
    output=tf.layers.max_pooling2d(x,k_size,strides=stride,padding='same')
    return output

def conv_trans(x,output_nums,conv_size,stride):
    output=tf.layers.conv2d_transpose(x,output_nums,conv_size,strides=stride,padding='same')
    return output

tf.reset_default_graph()
import tensorlayer as tl
log_file='./logs/train3'
Graph=tf.Graph()

with Graph.as_default():

    Input=tf.placeholder(tf.float32,[None,80,112,1])
    mask=tf.placeholder(tf.float32,[None,None,None,1])
    L=tf.placeholder(tf.float32,[None,1])
    with tf.name_scope('conv_3x3_0'):
        conv0=conv_2d(Input,16,[3,3],1)
        conv0_1=conv_2d(conv0,16,[3,3],1)
    with tf.name_scope('maxpool_0'):
        pool0=max_pool2d(conv0_1,[2,2],2)
    with tf.name_scope('conv_3x3_1'):
        conv1=conv_2d(pool0,32,[3,3],1)
        conv2=conv_2d(conv1,32,[3,3],1)
    with tf.name_scope('maxpool_1'):
        pool1=max_pool2d(conv2,[2,2],2)
    with tf.name_scope('conv_3x3_2'):
        conv3=conv_2d(pool1,64,[3,3],1)
        conv4=conv_2d(conv3,64,[3,3],1)
    with tf.name_scope('maxpool_2'):
        pool2=max_pool2d(conv4,[2,2],2)
    with tf.name_scope('conv_3x3_3'):
        conv5=conv_2d(pool2,128,[3,3],1)
        conv6=conv_2d(conv5,128,[3,3],1)
    with tf.name_scope('aux_out'):
        flat1=tf.layers.flatten(conv6)
        dense=tf.layers.dense(flat1,1)
        prob=tf.nn.sigmoid(dense)
    with tf.name_scope('upconv_1'):
        upconv1=conv_trans(conv6,64,[3,3],2)
    with tf.name_scope('concat_1'):
        concat_1=tf.concat([conv4,upconv1],3)
    with tf.name_scope('conv_3x3_4'):
        conv7=conv_2d(concat_1,64,[3,3],1)
        conv8=conv_2d(conv7,64,[3,3],1)
    with tf.name_scope('upconv_2'):
        upconv2=conv_trans(conv8,32,[3,3],2)
    with tf.name_scope('concat_2'):
        concat_2=tf.concat([conv2,upconv2],3)
    with tf.name_scope('conv_3x3_5'):
        conv9=conv_2d(concat_2,32,[3,3],1)
        conv10=conv_2d(conv9,32,[3,3],1)
    with tf.name_scope('upconv_3'):
        upconv3=conv_trans(conv10,16,[3,3],2)
    with tf.name_scope('concat_3'):
        concat_3=tf.concat([conv0_1,upconv3],3)
    with tf.name_scope('conv_3x3_6'):
        conv11=conv_2d(concat_3,16,[3,3],1)
        conv12=conv_2d(conv11,16,[3,3],1)
    with tf.name_scope('final_layer'):
        Final=conv_2d(conv12,1,[1,1],1,activation='sigmoid')
        # Final_mask=tf.cast(Final>=0.3,dtype=tf.float32)



    with tf.name_scope('loss'):
        ### Compute 1-(Dice Coeficient) as network's loss function
        aux_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=L,logits=dense))
        dice_loss=1-tl.cost.dice_coe(Final,tf.cast(mask,tf.float32),axis=[1,2,3])
        # Loss=dice_loss
        Loss=0.5*aux_loss+2*dice_loss
#        entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask,logits=Final)
#         loss=tf.reduce_mean(entropy)
        loss_summ=tf.summary.scalar('cost',Loss)

    optimizer=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9)
    train_op=optimizer.minimize(Loss)
    file_writer=tf.summary.FileWriter(log_file,Graph)
    init_op=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

if __name__=='__main__':
    labels=np.load('labels.pk')
    def get_batch(images,masks,labels,batch_size,seed=0):
        images_array=np.reshape(np.array(images),(len(images),1))
        masks_array=np.reshape(np.array(masks),(len(masks),1))
        np.random.seed(seed)
        shuffled_images=np.random.permutation(images_array)
        np.random.seed(seed)
        shuffled_masks=np.random.permutation(masks_array)
        np.random.seed(seed)
        shuffled_label=np.random.permutation(labels)
        for i in range(len(images)//batch_size):
            img=[]
            lab=[]
            sub_images=shuffled_images[i*batch_size:(i+1)*batch_size]
            sub_masks=shuffled_masks[i*batch_size:(i+1)*batch_size]
            sub_label=shuffled_label[i*batch_size:(i+1)*batch_size,:]
            for i in range(sub_images.shape[0]):
                imgs=cv2.imread(sub_images[i][0])
                labels=cv2.imread(sub_masks[i][0])
                imgs=imgs[:,:,0]
                labels=labels[:,:,0]
                imgs=cv2.resize(imgs,(112,80),interpolation=cv2.INTER_AREA)
                labels=cv2.resize(labels,(112,80),interpolation=cv2.INTER_AREA)
                img.append(imgs)
                lab.append(labels)
            yield np.stack(img),np.stack(lab),sub_label

    save_file='./image_segmentation/seg'
    num_epochs=10
    batch_size=64
    iter=0
    with tf.Session(graph=Graph) as sess:
        sess.run(init_op)
        for i in range(num_epochs):
            for img,lab,l in get_batch(train_images,train_mask,labels,batch_size,seed=i):
                iter+=1
                img=np.expand_dims(img,axis=-1)
                lab=np.expand_dims(lab,axis=-1)
                sess.run(train_op,feed_dict={Input:img/255.0,mask:lab/255.0,L:l})
                if iter%10==0:
                    summary_val=loss_summ.eval(feed_dict={Input:img/255.0,mask:lab/255.0,L:l})
                    file_writer.add_summary(summary_val,iter)
            current_loss=sess.run(Loss,feed_dict={Input:img/255.0,mask:lab/255.0,L:l})
            print('Epoch:{}/{}...... Loss: {}'.format(i+1,num_epochs,current_loss))
        Saver=tf.train.Saver()
        Saver.save(sess,save_file)
