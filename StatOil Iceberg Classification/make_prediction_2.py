import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
test_data=pd.read_json('test.json')
test_images=np.load('test_images.pk')
test_angle=np.load('test_angle.pk')
test_angle=np.expand_dims(test_angle,1)


def get_test_batch(images,angles,batch_size):
    num=images.shape[0]
    num_batch=np.ceil(num/batch_size)
    for i in range(int(num_batch)):
        if i<(int(num_batch-1)):
            image=images[i*batch_size:(i+1)*batch_size,:,:,:]
            angle=angles[i*batch_size:(i+1)*batch_size,:]
            yield image,angle
        else:
            image=images[i*batch_size:,:,:,:]
            angle=angles[i*batch_size:,:]
            yield image,angle


batch_size=100
loaded_graph=tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader=tf.train.import_meta_graph('./model_save_1/CK.meta')

    loader.restore(sess, tf.train.latest_checkpoint('./model_save_1/'))
    sig=loaded_graph.get_tensor_by_name('Logits/Sigmoid:0')
    image=loaded_graph.get_tensor_by_name('Placeholder:0')
    angle=loaded_graph.get_tensor_by_name('Placeholder_1:0')
    train_ind=loaded_graph.get_tensor_by_name('Placeholder_3:0')
    probs=[]
    for img,ang in get_test_batch(test_images,test_angle,batch_size):
        prob=sess.run(sig,feed_dict={image:img,
                                     angle:ang,
                                     train_ind:False})
        probs.append(prob)


result=np.concatenate(probs)
result=np.squeeze(result)
ID=test_data['id']
pickle.dump(result,open('result.pk','wb'))
submit=pd.DataFrame({'id':ID,'is_iceberg':result})
submit.to_csv('submit_21.csv',index=False)
