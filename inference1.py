from numpy import load
import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from bitstring import BitArray
import random
from random import randint
import matplotlib.pyplot as plt
import math
import tensorflow

from tensorflow.keras.models import model_from_json
import os

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))



# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

w = np.array(model.get_weights())
print (w)

model.set_weights(w)

# Predict on the 10000 test images.
predictions = model.predict(test_images[:10000])

#print (predictions)

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4, ....]

# Check our predictions against the ground truths.
print(test_labels[:10000]) # [7, 2, 1, 0, 4, ......]

x = np.argmax(predictions, axis=1)
y = test_labels[:10000]

count = 0
for i in range (0,10000):
	if x[i] == y[i]:
		count = count + 1
print ("accuracy = ", count / 10000)





