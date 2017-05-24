# Digit Recognizer using Deep Learning

Digit recognizer trained on MNIST dataset using TensorFlow 1.0 and Keras in Python 3.6.0. 

Data Source : http://yann.lecun.com/exdb/mnist/

References : 

https://www.tensorflow.org/get_started/mnist/pros

https://github.com/aymericdamien/TensorFlow-Examples

https://keras.io/getting-started/sequential-model-guide/

https://github.com/fchollet/keras/tree/master/examples


## What is in this repo

*mnist-tf-mlp.py*   Multilayer Perceptron
*mnist-tf-cnn.py*   Convolutional Neural Net
*mnist-tf-rnn.py*   Recurrent Neural Net

* Implementation of deep neural nets are trained against 55000 handwritten digit images using TensorFlow 1.0.
* Evluation is made on the test set of 10000 handwritten digit images.


*mnist-keras-mlp.py*   Multilayer Perceptron
*mnist-keras-cnn.py*   Convolutional Neural Net
*mnist-keras-rnn.py*   Recurrent Neural Net

* Implementation of deep neural nets are trained against 60000 handwritten digit images using Keras.
* Evluation is made on the test set of 10000 handwritten digit images.



## Model Evaluation
| Methods                         |  Accuracy (TensorFlow)  |  Accuracy (Keras)  |
| ------------------------------- |-------------------------|-------------------:|
| Multi-layer perceptrons         |  94.4.%                 |
| Convolutional neural network    |  98.0.%                 |
| Recurrent neural network        |  97.6.%                 |

