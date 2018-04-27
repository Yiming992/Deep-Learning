import tensorflow as tf
import numpy as np
import glob
import cv2
import os

def sort_test_files(s):
    base=os.path.basename(s)
    num=int(base[:-4])
    return num

if __name__=='__main__':

    mask_dir='\\test_mask'
    current_wd=os.getcwd()

    test_images=glob.glob('./test1/*.tif')
    test_images.sort(key=sort_test_files)


    loaded_graph=tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        loader=tf.train.import_meta_graph('./image_segmentation/seg.meta')
        loader.restore(sess,tf.train.latest_checkpoint('./image_segmentation/'))
        X=loaded_graph.get_tensor_by_name('Placeholder:0')
        mask=loaded_graph.get_tensor_by_name('final_layer/Sigmoid:0')
        for i in range(len(test_images)):
            test_img=cv2.imread(test_images[i])
            test_img=test_img[:,:,0]
            resized_img=cv2.resize(test_img,(112,80),interpolation=cv2.INTER_AREA)
            resized_img=resized_img[np.newaxis,:,:,np.newaxis]
            Mask=sess.run(mask,feed_dict={X:resized_img/255.0})
            Mask=np.squeeze(Mask)
            Mask=cv2.resize(Mask,(580,420),interpolation=cv2.INTER_AREA)
            Mask=Mask>0.5
            Mask=Mask*255.0
            # cv2.imshow('{}'.format(i),Mask)
            if not os.path.isdir(os.path.join(current_wd+mask_dir)):
                os.mkdir(os.path.join(current_wd+mask_dir))
            cv2.imwrite(os.path.join(current_wd+mask_dir+'\\{}_mask.png'.format(i+1)),Mask)
