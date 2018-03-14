import tensorflow as tf
import pickle
import numpy as np

train_images=np.load('train_images.pk')
train_label=np.load('train_label.pk')
train_angle=np.load('train_angle.pk')

def get_batch(images,angles,labels,batch_size):
    num=images.shape[0]
    num_batch=np.ceil(num/batch_size)
    for i in range(int(num_batch)):
        if i<(int(num_batch-1)):
            image=images[i*batch_size:(i+1)*batch_size,:,:,:]
            label=labels[i*batch_size:(i+1)*batch_size,:]
            angle=angles[i*batch_size:(i+1)*batch_size,:]
            yield image,angle,label
        else:
            image=images[i*batch_size:,:,:,:]
            label=labels[i*batch_size:,:]
            angle=angles[i*batch_size:,:]
            yield image,angle,label

loaded_graph=tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader=tf.train.import_meta_graph('./model_save/CK.meta')

    loader.restore(sess, tf.train.latest_checkpoint('./model_save/'))
    Loss=loaded_graph.get_tensor_by_name('loss/Mean:0')
    image=loaded_graph.get_tensor_by_name('Placeholder:0')
    angle=loaded_graph.get_tensor_by_name('Placeholder_1:0')
    label=loaded_graph.get_tensor_by_name('Placeholder_2:0')
    train_ind=loaded_graph.get_tensor_by_name('Placeholder_3:0')
    Losses=[]
    for im,an,la in get_batch(train_images,train_angle,train_label,20):
        lo=sess.run(Loss,feed_dict={image:im,
                                    angle:an,
                                    label:la,
                                    train_ind:False})
        Losses.append(lo)
    print("Loss: {}".format(sum(Losses)/len(Losses)))
