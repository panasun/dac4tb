#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import absolute_import, division, print_function, unicode_literals

try:
  get_ipython().system('pip install tensorflow_datasets')
  get_ipython().system('pip install scikit-learn')
  get_ipython().system('pip install urllib3')
  get_ipython().system('pip install Pillow')
  get_ipython().system('pip install tensorflow')
  get_ipython().system('pip install tensorflow-gpu')
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass


# In[2]:


import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile


# In[3]:


tf.__version__


# # Data preprocessing

# In[4]:


dataset_name = "DDC Prison BLM"

# if not os.path.exists('Dataset'):
#     os.mkdir('Dataset')
    
# with zipfile.ZipFile(dataset_name + ".zip","r") as zip_ref:
#     zip_ref.extractall("Dataset")


# In[5]:


import pathlib


# In[6]:


data_dir = pathlib.Path("Dataset/" + dataset_name)
data_dir


# In[7]:


image_count = len(list(data_dir.glob('*/*.jpg')))
image_count


# In[8]:


CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
CLASS_NAMES


# In[9]:


normal = list(data_dir.glob('N/*'))


# In[10]:


for image_path in normal[:3]:
    display.display(Image.open(str(image_path)))


# In[11]:


image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[12]:


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
IMG_HEIGHT = 224
IMG_WIDTH = 224
SPLIT_WEIGHTS = (7.5, 1.5, 1.5)
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)


# In[13]:


list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))


# In[14]:


for f in list_ds.take(5):
    print(f.numpy())


# In[15]:


CLASS_NAMES


# In[16]:


def get_label(file_path):
  # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


# In[17]:


def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # img = tf.cast(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


# In[18]:


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# In[19]:


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# In[20]:


for image, label in labeled_ds.take(2):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# In[21]:


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
    if cache:
        if isinstance(cache, str):
              ds = ds.cache(cache)
        else:
              ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# In[22]:


train_ds = prepare_for_training(labeled_ds)


# In[23]:


train_size = int(0.7 * image_count)
validation_size = int(0.15 * image_count)
test_size = int(0.15 * image_count)

train_dataset = train_ds.take(train_size)
test_dataset = train_ds.skip(train_size)
validation_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)


# In[ ]:


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')


# In[ ]:


train_image_batch, train_label_batch = next(iter(train_dataset))
show_batch(train_image_batch.numpy(), train_label_batch.numpy())


# In[ ]:


validation_image_batch, validation_label_batch = next(iter(validation_dataset))
show_batch(validation_image_batch.numpy(), validation_label_batch.numpy())


# In[ ]:


test_image_batch, test_label_batch = next(iter(test_dataset))
show_batch(test_image_batch.numpy(), test_label_batch.numpy())


# # Training

# In[ ]:


train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE)
validation_batches = validation_dataset
test_batches = test_dataset


# In[ ]:


for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape


# In[ ]:


IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


# base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
#                 include_top=True, 
#                 weights='imagenet', 
#                 input_shape=IMG_SHAPE, 
#                 pooling=None, 
#                 classes=1000)


# In[ ]:


feature_batch = base_model(image_batch)
print(feature_batch.shape)


# In[ ]:


base_model.trainable = False


# In[ ]:


base_model.summary()


# In[ ]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# In[ ]:


#prediction_layer = tf.keras.layers.Dense(2,activation="softmax")
prediction_layer = tf.keras.layers.Dense(len(CLASS_NAMES),activation="softmax")

prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# In[ ]:


model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# In[ ]:


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
#               loss='binary_crossentropy',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


len(model.trainable_variables)


# In[ ]:


num_train = train_size
num_val = validation_size
num_test = test_size


# In[ ]:


num_train, num_val, num_test


# In[ ]:


initial_epochs = 30
steps_per_epoch = round(num_train)//BATCH_SIZE
test_steps = round(num_test)//BATCH_SIZE
validation_steps = round(num_val)//BATCH_SIZE

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)


# In[ ]:


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# In[ ]:


history = model.fit(train_batches,
                    epochs=initial_epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_batches,
                    validation_steps=validation_steps)


# # Adam

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# # RMSProp

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:


base_model.trainable = True


# In[ ]:


# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# In[ ]:


model.compile(loss='categorical_crossentropy',
              optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate/10),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

# history_fine = model.fit(train_batches,
#                          epochs=total_epochs,
#                          initial_epoch =  history.epoch[-1],
#                          validation_data=validation_batches,
#                          validation_steps=validation_steps)

history_fine = model.fit(train_batches,
                    epochs=total_epochs,
                    initial_epoch =  history.epoch[-1],
                    steps_per_epoch=steps_per_epoch,
                    validation_data=validation_batches,
                    validation_steps=validation_steps)


# In[ ]:


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# In[ ]:


get_ipython().system('mkdir -p saved_model')
model.save(f'saved_model/{dataset_name}')


# In[ ]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.2, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[ ]:





# In[ ]:




