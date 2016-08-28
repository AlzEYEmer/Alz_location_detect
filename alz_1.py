
# coding: utf-8

# In[ ]:

from __future__ import print_function
import numpy as np
import os
import sys
from scipy import ndimage
import skimage
import skimage.io
import skimage.transform
import vgg16
import tensorflow as tf
from six.moves import range
import math


# In[ ]:


def pr(image):
    img=image
    img=img/255.0
    assert  (0 <= img).all() and (img <= 1.0).all()
    if len(img.shape)==2:
        img.resize(img.shape[0],img.shape[1],1)
        img=np.repeat(img,3,2)
    elif img.shape[2]>3:
        img=img[:,:,0:3]
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge,:]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    image_data=resized_img 
    images = tf.placeholder("float", [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)
    with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        feed_dict = {images: image_data.reshape((1,224,224,3))}
       
       
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
    tf.reset_default_graph()
    del vgg.data_dict
    del vgg
 
    return prob


# In[ ]:

#a=np.random.random((224,224,3))*255
#print pr(a)


# In[ ]:

def cl(image_data,reader):
    dic={0:'basement',1:'bathroom',2:'bedroom',3:'dining_room',4:'kitchen',5:'living_room',
             6:'street_building'}
    hidden_units=[1024]
   
    batch_size=1
    IMAGE_PIXELS=1000
    NUM_CLASSES=len(dic.keys())
    graph=tf.Graph()
    with graph.as_default():
        tf_test_dataset= tf.placeholder("float", [1,1000])
        def inference(images1,hidden_units):
            l=0
            N=len(hidden_units)
            weights=[]
            biases=[]
            for h in range(N+1):
                a=IMAGE_PIXELS if h==0 else hidden_units[h-1]
                b=NUM_CLASSES if h==N else hidden_units[h]
                weights.append(tf.constant(reader.get_tensor('weights'+str(h)),
                    name='weights'+str(h)))
                biases.append(tf.Variable(tf.zeros([b]),name='biases'+str(h)))
                l=l+tf.nn.l2_loss(weights[h])
            input_1=images1
            for h in range(N):
                input_1= tf.nn.relu(tf.matmul(input_1, weights[h]) + biases[h])
            logits1 = tf.matmul(input_1, weights[N]) + biases[N]
            return logits1
        logits1=inference(tf_test_dataset,hidden_units)
        test_prediction=tf.nn.softmax(logits1)
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        #print ("Initialized")
        feed_dict={tf_test_dataset:image_data}
        predictions = session.run([test_prediction], feed_dict=feed_dict)
 
    if np.amax(predictions[0])>0.5 :
        return dic[np.argmax(predictions[0],1)[0]]
    else:
        return "it is not detected successfully"
    
    
            


# In[ ]:

'''a=np.random.random((1,1000))
a=a/sum(a)
b=cl(a)
print b'''


# In[ ]:

def vgg_cl(image_1):
    
    reader=tf.train.NewCheckpointReader('model.ckpt')
    image_1_data=pr(image_1)
    class_name=cl(image_1_data,reader)
    return class_name


# In[ ]:


#a=skimage.io.imread('IMG_0204.JPG')
#print (vgg_cl(a))

