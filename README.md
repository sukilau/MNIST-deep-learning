# Digit Recognizer using Deep Learning

Digit recognizer trained on MNIST dataset using TensorFlow 1.0 in Python 3.6.0. 

Data Source : http://yann.lecun.com/exdb/mnist/
References : 
https://www.tensorflow.org/get_started/mnist/pros
https://github.com/aymericdamien/TensorFlow-Examples


## What is in this repo

* Implementation of deep neural nets are trained against 55000 handwritten digit images.
* Evluation is made on the test set of 10000 handwritten digit images.


*mnist-tf-mlp.py*  

* Multilayer Perceptron
* 2 hidden layers with ReLu activation, linear output layer


*mnist-tf-cnn.py*  

* Multilayer convolutional neural network
* 2 convolution layers with ReLu activation and max pooling, fully connected layer with ReLu activation, dropout, linear output layer


*mnist-tf-rnn.py*  

* Recurrent neural network
* LSTM with 128 neurons using last output, linear output layer



## Model Evaluation
| Methods                         |  Accuracy   |
| ------------------------------- |------------:|
| Multi-layer perceptrons         |  94.4.%     |
| Convolutional neural network    |  98.0.%     |
| Recurrent neural network        |  97.6.%     |

