#### To Study Implementation of Gradient Descent for Multi-class Classification Using a SoftMax Regression and Neural Networks

##### Ashutosh Singh         Operations Research and Industrial Engineering
##### Yanxin Li              Department of Statistics and Data Sciences

This is our final project for the Fall 2017 class SDS 385: Statistical Models for Big Data taught by Professor James Scott at the University of Texas at Austin.

####  Introduction
- We adopted a step-wise approach and started with SoftMax regression on dataset with 1010 classes and then extended the work to accommodate multiple layers.

- We studied the mathematics associated with backpropagation and programmed a function that can take in a list of predictors, weights and biases and returns cross-entropy loss, modified weights and modified biases.

- The dimension and initial choices for the weights and biases depends on the design choice and must be decided by the user.

- The function is:
Neural\_network(X, y, weight, bias, step\_size,  activation)

Where,

>* X= predictors.
>* y= one hot encoded label.
>* weight= list of weights for each layer, in increasing order of layer index.
>* bias= list of bias for each layer, in increasing order of layer index.
>* step_size= step size for gradient descent.
>* activation=0 for sigmoid, 11 for ReLU (Rectified Linear Unit).

Return,

>* Loss= cross entropy loss.
>* weight= list of modified weights for each layer, in increasing order of layer index.
>* bias= list of modified bias for each layer, in increasing order of layer index.

- The code is highly modular, and the reader can play around to create variations.
- This function can be set on a loop to fit simple Neural Networks with sufficient accuracy.
- We wrote a feed function to provide mini batches to the loop.
- The data was used after PCA transformation and dropping unimportant predictors.
- We programmed our function using **MNIST** data set and we cross checked the function by fitting various Neural Networks configurations on different models.
- mWe have presented the work for one such dataset, called **notMNIST**. The dataset is like MNIST with 10 classes of English alphabets from A to J.


