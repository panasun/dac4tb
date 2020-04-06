#!/usr/bin/env python
# coding: utf-8

# In[215]:


from __future__ import absolute_import, division, print_function, unicode_literals


# In[216]:


# try:
#   !pip install tensorflow
# except Exception:
#   pass


# In[217]:


import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
# import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import pathlib
import datetime
import warnings


# In[218]:


tf.__version__


# In[219]:


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[220]:


np.random.seed(1337) 
tf.random.set_seed(1337)


# In[221]:


data_dir = pathlib.Path("input")
data_dir


# In[222]:


image_count = len(list(data_dir.glob('*.jpg')))
image_count


# In[223]:


print("START PREDICTION: " + str(data_dir))
print("TOTAL IMAGES: " + str(image_count))


# In[224]:


images = list(data_dir.glob('*'))


# In[225]:


# for image_path in images[:3]:
#     display.display(Image.open(str(image_path)))


# In[226]:


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[227]:


SHUFFLE_BUFFER_SIZE = 1000
IMG_HEIGHT = 224
IMG_WIDTH = 224


# In[228]:


list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'))


# In[229]:


# for f in list_ds.take(5):
#     print(f.numpy())


# In[230]:


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # img = tf.cast(img, tf.float32)
  # resize the image to the desired size.
  return [tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])]


# In[231]:


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img , file_path


# In[232]:


prep_ds = list_ds.map(process_path)
prep_ds = prep_ds.take(image_count)
# print((prep_ds))


# In[233]:


CLASS_NAMES = ['ABNORMAL', 'NORMAL']
MODEL_NAME = 'blm_2_t_mn'
print("MODEL NAME: " + MODEL_NAME)
predict_model = tf.keras.models.load_model('saved_model/' + MODEL_NAME + '.h5')
predict_model.summary()


# In[234]:


with tf.device('/cpu:0'):
    predictions = predict_model.predict(prep_ds)


# In[235]:


currentDT = datetime.datetime.now()
currentDT = currentDT.strftime("%Y%m%d%H%M%S")
outputFile = "output/output_" + currentDT + ".csv"
f = open(outputFile, "w")


# In[236]:


f.write('IMAGE' + ',' + 'MODEL_NAME' + ',')
for r in CLASS_NAMES:
    f.write(r + ',')
f.write('PREDICT' + '\n')


# In[237]:


p = 0
for ds in list_ds:
    img_path = str(os.path.splitext(ds.numpy())[0])
    img_path = img_path.replace("b", "")
    img_path = img_path.replace("\\", "")
    img_path = img_path.replace("/", "")
    img_path = img_path.replace("input", "")
    img_path = img_path.replace("'", "")

    res = predictions[p]
    msg = ""
    
    f.write(img_path + ',' + MODEL_NAME + ',')
    msg += str(p) + '\t' + img_path + '\t' + MODEL_NAME + '\t'
    pred = 0
    idx = 0
    pred_score = 0
    
    
    for r in res:
        f.write(str(r) + ',')
        msg += str(r) + '\t'
        if r > pred_score:
            pred_score = r
            pred = idx
        idx = idx + 1
        
    f.write(CLASS_NAMES[pred] + '\n')
    msg += CLASS_NAMES[pred]
    print(msg)
    
    p = p + 1


# In[238]:


f.close()


# In[ ]:


print("OUTPUT FILE: " + outputFile)

