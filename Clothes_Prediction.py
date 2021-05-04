!pip install -U tensorflow_datasets

from __future__ import absolute_import,division,print_function

import tensorflow as tf
import tensorflow_datasets as tfds

#helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

#for progress bar
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

print(tf.__version__)

dataset,metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


class_names = ['T-shirts/top', 'Trouser', 'pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Explore Data
#print(metadata)
num_train_ex = metadata.splits['train'].num_examples
num_test_ex = metadata.splits['test'].num_examples
print(num_train_ex, num_test_ex)

# preprocess data
def normalize(images,labels):
  images = tf.cast(images,tf.float32)
  images /= 255
  return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#exploere processed data
for image, label in test_dataset.take(1):
  break

image = image.numpy().reshape((28,28))

plt.figure
plt.imshow(image, cmap = plt.cm.binary)

plt.colorbar()
plt.grid(False)
plt.show()

# take look at first 25 cloths
plt.figure(figsize=(10,10))

i=0
for (image,label) in test_dataset.take(25):
  plt.grid(False)
  image = image.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(image, cmap=plt.cm.binary)
  plt.xlabel(class_names[label])
  i+=1
plt.show()

#Build the model
#1) setup the layers

model = tf.keras.Sequential(
    [
     tf.keras.layers.Flatten(input_shape=(28,28,1)),
     tf.keras.layers.Dense(units = 128,activation=tf.nn.relu),
     tf.keras.layers.Dense(units = 10, activation=tf.nn.softmax)
     ]
)

#2) compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


#Train the model

BATCH_SIZE = 32

train_dataset = train_dataset.repeat().shuffle(num_train_ex).batch(BATCH_SIZE)

test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=6,  steps_per_epoch=math.ceil(num_train_ex/BATCH_SIZE))

print("\n trained MODEL")

#Evalute Accuracy
test_loss,test_accuracy = model.evaluate(test_dataset,steps=math.ceil(num_test_ex/32))

print(test_accuracy)
print(test_loss)

#make prediction and explore

for test_images,test_labels in test_dataset.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()

  prediction = model.predict(test_images)

prediction[0]

prediction.shape

test_labels[0]

np.argmax(prediction[0])

class_names[np.argmax(prediction[0])]

print(len(prediction))

#lets see whether its coat or not

def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  # print(img)
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction, test_labels)

# now time to see 
i = 4
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction, test_labels)

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, prediction, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, prediction, test_labels)

# serveral images at a time 

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, prediction, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, prediction, test_labels)

#Finally, use the trained model to make a prediction about a single image. 
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = np.array([img])

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

class_names[np.argmax(predictions_single[0])]
