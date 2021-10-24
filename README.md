# ARA
In this project, we create and implement a deep learning library from scratch. 


<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Deep Leaning Library](#deep-leaning-library)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
    - [Aim](#aim)
    - [Tech Stack](#tech-stack)
    - [File Structure](#file-structure)
  - [Approach](#approach)
  - [Theory](#theory)
      - [Loss Function:](#loss-function)
      - [Cost Function :](#cost-function-)
      - [Gradient Descent : -](#gradient-descent---)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Results](#results)
    - [Result](#result)
  - [Future Work](#future-work)
  - [Troubleshooting](#troubleshooting)
  - [Contributors](#contributors)
  - [Acknowledgements](#acknowledgements)
  - [Resources](#resources)
  - [License](#license)
  *[Results](#Result)
  *[Demo Video](#Demo)
* [Future Work](#future-work)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [Acknowledgements](#acknowledgements)
* [Resources](#resources)
* [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project

>Deep learning can be considered as a subset of machine learning. It is a field that is based on learning and improving on its own by examining computer algorithms. Deep learning works with artificial neural networks consisting of many layers.
This project, which is creating a Deep Learning Library from scratch, can be further implemented in various kinds of projects that involve Deep Learning. Which include, but are not limited to applications in Image, Natural Language and Speech processing, among others.

### Aim
>To implement a deep learning library from scratch.

### Tech Stack
>Technologies used in the project:
>* Python and numpy, pandas, matplotlib
>* Google Colab

### File Structure

```
.
├── code
|   └── main.py                                   #contains the main code for the library
├── resources                                     #Notes 
|   ├── ImprovingDeepNeuralNetworks
|   |   ├── images
|   |   |   ├── BatchvsMiniBatch.png
|   |   |   ├── Bias.png
|   |   |   └── EWG.png
|   |   └── notes.md
|   ├── Course1.md                               
|   ├── accuracy.jpg
|   ├── error.jpg
|   └── grad_des_graph.jpg
├── LICENSE.txt
├── ProjectReport.pdf                            #Project Report
└── README.md                                    #Readme
```
    
## Approach
>The approach of the project is to basically create a deep learning library, as stated before. The aim of the project was to implement various deep learning algorithms, in order to drive a deep neural network and hence,create a deep learning library, which is modular,and driven on user input so that it can be applied for various deep learning processes, and to train and test it against a model.

## Theory
>A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.
>
>There are different types of Neural Networks
>
>* Standard Neural Networks
>* Convolutional Neural Networks 
>* Recurring Neural Networks 

<!-- In Deep Learning, a neural network with multiple layers, or a deep neural network is applied. A deep learning process is gauged by both the performance of the neural network, as well as the amount of data involved in the process.
With the same amount of data used for training, the performance of the Neural Network rises with Learning Algorithm or type of NN used. -->
#### Loss Function:
>Loss function is defined so as to see how good the output ŷ is compared to output label y.
![](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+L%28%5Chat+y%2C+y%29+%3D+-%28ylog+%5Chat+y+%2B+%281-y%29log%281-%5Chat+y%29%29)


#### Cost Function : 
>Cost Function quantifies the error between predicted values and expected values.
![](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+J%28w%2Cb%29+%3D+%5Cfrac%7B1%7D%7Bm%7D++%5Csum_%7Bi%3D1%7D%5Em+L%28%5Chat+y%5E%7B%28i%29%7D+%2C+y%5E%7B%28i%29%7D%29+%3D+-%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+y%5E%7B%28i%29%7Dlog%28%5Chat+y%5E%7B%28i%29%7D%29%2B%281-y%5E%7B%28i%29%7Dlog%281-%5Chat+y%5E%7B%28i%29%7D%29%29)

#### Gradient Descent : -
>Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. 





<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Object oriented programming in Python 

* Linear Algebra 

* Basic knowledge of Neural Networks
<!-- ```sh
How to install them
``` -->

* **Python 3.6 and above**

  You can visit the [Python Download Guide](https://www.python.org/downloads/) for the installation steps.
  
* Install numpy next  
```
pip install numpy
```

### Installation
1. Clone the repo
```sh
git clone git@github.com:aayushmehta123/sra_eklavya_deeplearning_library.git
```


<!-- USAGE EXAMPLES -->
<!-- ## Usage
```
Deep learning library is used train the models more easily. 

``` -->


<!-- RESULTS AND DEMO -->
## Results 

### Result
Results obtained during training:
![error](/resources/error.jpg)
(where Y-axis represents the value of the cost function and X axis represents the number of iterations)
![accuracy](/resources/accuracy.jpg)
(where Y-axis represents the accuracy of the prediction wrt the labels and X-axis represents the number of iterations)
<!-- FUTURE WORK -->
## Future Work

* Short term
  * Adding class for normalization and regularization
* Near Future
  * Addition of support for linear regression
  * Addition of classes for LSTM and GRU blocks
* Future goal
  * Addition of algorithms to support CNN models. 
  * Addition of more Machine Learning algorithms
  * Include algorithms to facilitate Image Recognition, Machine Translation and Natural Language Processing


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Numpy library not working so we shifted workspace to colab


<!-- CONTRIBUTORS -->
## Contributors
* [Rishabh Bali](https://github.com/rishabh2002-lang)
* [Aditya Mhatre](https://github.com/Adi935)
* [Aayush Mehta](https://github.com/aayushmehta123)


<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements
* [SRA VJTI](http://sra.vjti.info/) Eklavya 2021  
* Mentors: 
  * [Kush Kothari](https://github.com/kkothari2001) 
  * [Aman Chhaparia](https://github.com/amanchhaparia)

## Resources
* [Coursera course by Andrew NG on Deep Learning and Neural Networks](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome)
* [Coursera course by Andrew NG on Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)
* [On Gradient Descent Optimization](https://ruder.io/optimizing-gradient-descent/)
* [Essence of Linear Algebra by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
* [Basics of Neural Networks by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [Basics of Implementing Deep Learning Libraries](https://towardsdatascience.com/on-implementing-deep-learning-library-from-scratch-in-python-c93c942710a8)
* [Deep Learning](https://www.ibm.com/cloud/learn/deep-learning)
* [Deep Learning Libraries](https://www.sabinasz.net/deep-learning-from-scratch-theory-and-implementation/) 
* [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)



<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project. 
