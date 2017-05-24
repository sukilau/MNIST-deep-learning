# MLP 


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


# set parameters
batch_size = 128
epochs = 10
n_classes = 10

# load MNIST data 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# view image
for i in range(6,9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);

# data preprocessing
x_train = x_train.reshape(60000, 784)  # reshape input from (28,28) to 784
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

scale = np.max(x_train) # 255
x_train /= scale
x_test /= scale

mean = np.std(x_train)
x_train -= mean
x_test -= mean

input_dim = x_train.shape[1]
print('x_train shape:', x_train.shape)  # (60000, 784)
print(x_train.shape[0], 'train samples') # 60000
print(x_test.shape[0], 'test samples')  # 10000

y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

# construct MLP
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=input_dim))  # imput_dim = 784
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer="rmsprop",metrics=['accuracy'])

# train MLP
model.summary()
history = model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

# evaluate on test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# visualize the loss function in each epoch
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo') 
plt.plot(epochs, val_loss_values, 'b+') 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# visualize accuracy in each epoch
plt.clf() 
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()