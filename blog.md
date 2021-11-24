# Eklavya : Deep Learning Library from Scratch
___
## Introduction
___

Eklavya is a program conducted by the [Society of Robotics and Automation, VJTI](https://github.com/SRA-VJTI) in which freshmen study and execute various projects amongst various fields in ML, Embedded Systems, amongst various others under the guidance of their seniors as their mentors.

The Project that was completed by our team ([Rishabh Bali](https://github.com/Ris-Bali), [Aayush Mehta](https://github.com/AayushM8) and [Aditya Mhatre](https://github.com/Adi935)) , was understanding and making a **Deep Learning Library**, from scratch. We learnt about various aspects and basic concepts of Deep Learning.

___

### What is Deep Learning ?

Let's get formal while defining deep learning - Deep learning is a subset of machine learning which structures inspired form our brain cells called artificial neural networks.
It's basically a type of machine learning that trains a computer to perform human-like tasks, such as recognizing speech, identifying images or making predictions. So deep learning is a subset of machine learning in which we train neural network (the computer program) over many layers using equations with parameters which are updated after every iteration which makes these networks to be very good at recognizing patterns. In the era of big data where we literally have an infinite amount of data deep learning is blooming as its accuracy of making predictions increases with the amount of data . These deep Neural nets are used to perform various human tasks with human level accuracy or higher .Like detecting a Tumor with the help of intelligence . 
Deep Learning has distinct advantages over traditional ML algorithms . AS it gives more accurate results . 

![/assets/1.jpg](https://github.com/Ris-Bali/ARA/blob/main/assets/1.jpg?raw=true)
___
### What is a deep learning library and the reason why we chose this project ?

A library is defined as a collection of pre-defined functions and modules that you can call through your program .These Libraries are primarily open source and are maintained by large communities. Deep Learning is a subset of Machine Learning in which we build Multi Layer Neural Networks (consider this as a tree) which progressively extracts higher level features from data. More informally we try to build a mathematical function to map the inputs to the outputs.

There are primarily two main layers the input layer and the output layer and the hidden layers are kinda sandwiched between the two every layer except for the input layer may or may not have an activation function. Building these layers require a lot of code; consider this most of the industry standard neural networks have a at least 8 to 20 layers. Oh, and also every layer has nodes, and every node has its set of parameters so this leads a very complex structure which is very difficult to code. Deep Learning libraries automate this process and save a ton of time and drastically reduces the time required to build a deep net.

In the case of deep learning there are many libraries that make the process of coding deep neural nets easier. Most notable are Tensorflow (By Google) and Pytorch (By Facebook) while newer libraries like fast.ai ‘ deep learning library are gaining traction. When I got selected in Eklavya which is SRA(Society of Robotics and Automation)’s mentorship program where seniors mentor us while we build our projects. I was confused between two projects Deep Learning Library and RL agents I chose the former after consulting with a senior as this would allow me to get my hands dirty. I would get to implement all the algorithms from scratch; this is beneficial if you want to get into research. As you would see in the latter part of the article most of the frameworks (which a fancy term for a Deep Learning Library with readymade implementation of the forward propagation and backward propagation part. The Forward propagation part is easy enough to understand but the backward propagation is the part which involves a lot of calculus. Deep Learning is not only used by people from engineering background but by Medical practitioners, financial planners and so on who don't have any formal background in Maths (at least the calculus part). It is possible to build models without bothering to know the back propagation part. I thought it would be fun to know what happens behind the scenes so I chose this project. Teams were formed, and I got grouped with two other members. 

___
### The Language problem:
So the first problem in front of us was the language to use for this library. Tensorflow and Pytorch are written in C++ and CUDA (Nvidia's language for programming) and modelled to have a syntax like python. These libraries have C++ at their core (like most gaming engines) because C++ is fast and we need that speed because training Deep Learning models has a lot of computationally expensive steps which makes it mandatory to have high performance. But C++ is syntactically difficult to code and unlike some famous interpreted languages do not have many helpful libraries like numpy, Pandas, Scipy, Matplotlib, etc. So we decided to use Python 
___

### Prerequisites: 
Our mentors informed us we would need a sound foundation of Linear Algebra and OOP in Python before starting the project. So we were instructed to watch 
1) 3b1b's Video Lectures on Linear Algebra
2) Corey Schafer's playlist on Python

We also learnt about Deep Learning from the Deep Learning Specialization course offered by Andrew N.G. on Coursera.

___

### The Library: 

Once we had the knowledge of all the algorithms, it was time to implement it in our library. First let's describe the process and the algorithms we used.
 
1) Forward Propagation step - In this step we make predictions using the data provided as input the data is divide into two parts inputs (X) and the output labels (usually denoted by Y) and the predictions we make are denoted by ŷ
where, ***ŷ= W<sup>T</sup> X +B***

2) The cost function - We need to make as accurate predictions as possible the predicted yhat value should be close to Y. We can make use of the standard loss function or the cross entropy loss function. The loss function summed over M training examples is called the cost function. (Note - There is a concept called vectorization - which is the process of converting the m input vectors and output labels into vectors as this makes the computation faster) 

3) The Optimizers (Optimizing Algorithms) - We need to minimize the loss(cost function),in other words we need to manipulate the parameters to their optimum values. This can be achieved by using the optimizing algorithms . 
The Important algorithms are -
   * The Stochastic Gradient Descent Algorithm 
   * Adam Optimization Algorithm
   * RMS Prop Algorithm
We have implemented the SGD algorithm in this library 
___

### Classes in The Library

So our first hurdle was to find a way to allow the user to manually set activation functions for every layer and also set the weights and biases for every layer (parameters)and the activations.
1) **Initialization class**: We found the solution to the above problem by creating empty dictionaries for each of the above components. And randomly assign weights and biases to every layer.
![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/2.jpg?raw=true)
2) **Activation class**: In this class we defined the Sigmoid and Tanh activation functions.
![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/3.jpg?raw=true)
3) **Forward prop class**: In this class we calculate the predictions and the cost ; using the weights and biases and the activations dictionaries. We make use of f strings and create keys for the dictionary.
![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/4.jpg?raw=true)
4) **Activation backward class**: In this class we have defined the derivative of the cost function with respect to the activation functions.
![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/5.jpg?raw=true)
5) **Backward prop class**:  In this class we have implemented the back prop formula to compute dw and db the derivative of the cost function wrt to the Cost function
![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/6.jpg?raw=true)
6) **Updation Class** - in this class we update the parameters so as to reduce the cost function to its minimum.
![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/7.jpg?raw=true)
___
### Results Obtained 

We trained a model with the help of our Deep learning library using the [Iris Dataset](https://www.kaggle.com/uciml/iris)

and we obtained the following results

![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/8.jpg?raw=true)

![bruh](https://github.com/Ris-Bali/ARA/blob/main/assets/9.jpg?raw=true)
___
The library is in no way complete the project has no end to it currently the library can perform only binary and multiclass classification using the SGD algorithm we need to add the Adam Optimization, and the RMS prop algorithms which couldn't be added because of time constraints.We even need to add the regularizers which prevent the model from over fitting the training set and allows the model to generalize to give highly accurate predictions on the test set. We need to add the algorithms used in Computer Vision and sequence models.

So it will surely be an exciting journey ahead.


