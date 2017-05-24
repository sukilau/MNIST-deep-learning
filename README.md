# Digit Recognizer using Deep Learning

Digit recognizer trained on MNIST dataset using TensorFlow 1.0 and Keras in Python 3.6.0. 

Data Source : http://yann.lecun.com/exdb/mnist/

References : 

https://www.tensorflow.org/get_started/mnist/pros

https://github.com/aymericdamien/TensorFlow-Examples

https://keras.io/getting-started/sequential-model-guide/

https://github.com/fchollet/keras/tree/master/examples


## What is in this repo

*mnist-tf-mlp.py*, *mnist-tf-cnn.py*, *mnist-tf-rnn.py*

* Implementation of deep neural nets, ie. multilayer perceptrons, convoluntional neural net and recurrent neural net, are trained against 55000 handwritten digit images using TensorFlow 1.0.
* Evluation is made on the test set of 10000 handwritten digit images.


*mnist-keras-mlp.py*, *mnist-keras-cnn.py*, *mnist-keras-rnn.py*

* Implementation of deep neural nets, ie. multilayer perceptrons, convoluntional neural net and recurrent neural net, are trained against 60000 handwritten digit images using Keras.
* Evluation is made on the test set of 10000 handwritten digit images.



## Model Evaluation
| Methods                         |  Test accuracy (TensorFlow)  |  Test accuracy (Keras)  |
| ------------------------------- |------------------------------|------------------------:|
| Multi-layer perceptrons         |  94.4.%                      |  97.6%                  |
| Convolutional neural network    |  98.0.%                      |  xx.x%                  |
| Recurrent neural network        |  97.6.%                      |  xx.x%                  |

