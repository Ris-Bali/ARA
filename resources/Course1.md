# Neural Networks and Deep Learning [Coursera] https://www.coursera.org/learn/neural-networks-deep-learning/
---
## Introduction to Deep Learning 
---
### What is a Neural Network?
 A neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes. Artificial Neurons sometimes inputs the size, computes this linear function,takes a max of zero, and then outputs the estimated price. This is called famously as RELU. 
###### F/N :- ReLU - Rectified Linear Unit Function
#### For Example
For Housing Size Prediction models, we have data sets of houses, the area, the buyers, and then use it as a function to predict the price of the houses

![HousingExample](https://i.ibb.co/X3zHZtK/Neural-Network-Eg-1.png)

![HousingExample](https://i.ibb.co/LRSTQXC/Neural-Network-Eg-2.png)

So, in a Neural Network, there are a lot of Input Layers, Output Layers, and Hidden Layers (Which are actually neurons, but we can call them layers as they're collection of multiple neurons working together )
Also, the hidden layers and input layers are interdependant on each other

---
 ### Supervised learning with Neural Networks

 | Input(x)        |Output(y)       |Application |
 |-----------------|:--------------:|-----------:|
 |Home Features|Price| Real Estate |
 |Ad, User info|Click on ad? (0/1)| Online Advertising|
 |Image|Object(0,1,...,10000)|Photo tagging|
 |Audio|Text Transcript|Speech Recognition|
 |English|Japanese|Machine Translation|
 |Image, Radar Info| Position of Other Cars| Autonomous Driving|

 These are a few applications for Supervised Learning w/ NN.

![NeuralNetworkExamples](https://i.ibb.co/NnCRBJz/Neural-Network.png)

 #### There are different types of Neural Networks

 1. **Standard Neural Networks**
    The normal type of Neural Networks with a groups of multiple neurons at each layer, it's only processed in the forward direction.
    It consists of 3 layers, Input, Hidden and Output
    It is used for Simpler problems.
 2. **Convolutional Neural Networks**
    Convolutional Neural Networks work on the principle of Convolution Kernels and is most commonly used for Image processing

    The process of convolution is applied in these types of networks, which states that `for a mathematical operation
    on two functions (f and g) that produces a third function (f∗g) that expresses how the shape of one is modified by the other.`

 3. **Recurring Neural Networks**
    RNNs are Neural Networks which are a derivative of Standard Neural Networks in which, a looping constraint on the hidden layer of the SNN turns it into an RNN.
    This type of NN is used for one-dimensional tempral sequence or multi-dimensional sequence data, like Audio, Video.
 4. **Hybrid Neural Networks**
    These involve complex problems like Autonomous Car Systems which use different types of NN in sync with each other.

### Types of Data

1. **Structured Data**
    Each of the features such as Age, Size etc, have well defined meanings

2. **Unstructred Data**
    Things like Audio, Image, Text, which is unstructured, and vaguely random.
    Things like Speech Recognition, Image Recognition, Natural Language Processing are required for Unstructed Data analysis.

### Scale of DL Progress

![DLProgress](https://i.ibb.co/gVp93bp/DL-progress.png)

For a Deep Learning Process, we take a look at both Performance and Amount of Data involved.

With the same amount of data used for training, the perfomance of the Neural Network rises with Learning Algorithm or type of NN used

For example, a Large NN has higher performance than a Medium or a Small NN, and even more so than a Traditional Algorithm for DL.

Also, in a smaller training set, this scale is relatively similar and random
 ---
   (Week 2)
   ---
   ## Introduction to Deep Learning
   ---

   ### Binary Classification

   Logistic Regression is an algorithm for binary classification.

   **For Example** :- Binary Classification in an Image

   In our Computers, an image is usually stored in the format of three matrices, each of which corresponds to the Red, Green and Blue (RGB) color channel.
   Depending on the pixels of the original image, say, 128 pixels, we would have 3 matrices corresponding to RGB Pixel Intensity Values of that image of the size 128 x 128.

   So, to define it as a Feature Vector, we take all the pixel values from the image, take a Vector **X** and then start listing the values, until we have listed all Red pixels, followed by Green pixels and then finally Blue pixels, and so on, till we get a long feature vector listing all of these values of that image.

   The Image is 128 x 128 pixels, hence the total dimension of this feature vector is *128 x 128 x 3 =49512*

   Hence, n = n<sub>x</sub> = 49512

   So, if we have a given image, and have to identify it as a train or a non train image, for which we take y = 1 (train image) or y = 0 (non train image), we have to get an output as 1 or 0 from the given Feature Vector.

   ##### Notation used

   (x, y) where x E R<sup>n<sub>x</sup></sub> and y ∈ {0 , 1}

   Training sets are **m** 
   Training Examples are  {(x<sup>(1)</sup> , y<sup>(1)</sup>) , (x<sup>(2)</sup> , y<sup>(2)</sup>),...(x<sup>(m)</sup> , y<sup>(m)</sup>)}

   **m** = **m<sub>train</sub>**
   **m<sub>test<sub>** = Number of Test Examples

   For Simplicity, we define a Vector **X**

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+X+%3D%5Cbegin%7Bbmatrix%7D+%7C%26+%7C%26+%7C%26+%7C%26+%7C%26+%7C+%5C%5C%5C+x%5E%281%29+%26+x%5E%282%29+%26+.+%26+.+%26+.+%26+x%5E%28m%29+%5C%5C%5C+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C%5Cend%7Bbmatrix%7D+%0A%0A%0A" >

   where, it has m columns and n<sub>x</sub> rows

   **X** ∈ R<sup>n<sub>x</sub>  x m </sub>

   **For python**

   ` X.shape = (n_x, m)`

   Similarly, 

   **Y** = **X** = 

   **Y** ∈ R<sup>1  x m </sub>
   **For python**

   `Y.shape = (1, m)`

   ---

   ### Logistic Regression

   Given x, output prediction **ŷ = P(y=1 | x)**

   **x** ∈ R<sup>n<sub>x</sup></sub>
   also, by Parameters, w ∈ R<sup>n<sub>x</sup></sub> and b ∈ R

   Output being **ŷ = w<sup>T</sup>x + b**
   where w<sup>T</sup>x is transpose, this is linear regression equation

   but this isn't good for binary classification, as **0 ≤ ŷ ≤ 1** as ŷ = w<sup>T</sup>x+b can be > 1 or < 0

   Hemce, we take output as **ŷ** = **σ(w<sup>T</sup>x + b)** = **z**

   where σ is the *sigmoid* function, which is defined as

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csigma%28z%29+%3D+%5Cfrac%7B1%7D%7B1%2Be%5E%7B%28-z%29%7D+%7D" 
alt="\sigma(z) = \frac{1}{1+e^{(-z)} }">
   **and if z is large**
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csigma%28z%29+%5Capprox+%5Cfrac%7B1%7D%7B1%2B0%7D++%3D+1" 
alt="\sigma(z) \approx \frac{1}{1+0}  = 1">
   **if z is negatively large**
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Csigma%28z%29+%5Capprox+%5Cfrac%7B1%7D%7B1%2BLargeNumber%7D++%3D+0" 
alt="\sigma(z) \approx \frac{1}{1+LargeNumber}  = 0">
   and graph is

   ![SigmoidGraph](https://miro.medium.com/max/800/1*OUOB_YF41M-O4GgZH_F2rw.png)

   ###### In an alternative notation, 

   x<sub>0</sub> = 1 , **x** ∈ R<sup>n<sub>x+1</sup></sub>

   and **ŷ** = **σ(θ<sup>T</sup>x)**

 <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Ctheta+%3D+%5Cbegin%7Bbmatrix%7D+%5Ctheta%5E%280%29+%5C%5C+%5Ctheta%5E%281%29+%5C%5C+%5Ctheta%5E%282%29+%5C%5C+.+%5C%5C+.+%5C%5C+.+%5C%5C%5Ctheta_%28n_x%29%5Cend%7Bbmatrix%7D" >

   where, θ<sup>0</sup> functions **b** and the rest from θ<sup>1</sup> to θ<sub>n<sub>x</sub></sub> function as **w**

   #### **Logistic Regression Cost Function**

   To train the models of **w** and **b** we use the Logistic Regression _cost_ function

   Given : {(x<sup>(1)</sup> , y<sup>(1)</sup>) , (x<sup>(2)</sup> , y<sup>(2)</sup>),...(x<sup>(m)</sup> , y<sup>(m)</sup>)} and want : ŷ<sup>(i)</sup> ≈ y<sup>(i)</sup>

   where, **(i)** , comes from

   **ŷ<sup>(i)</sup>** = **σ(w<sup>T</sup>x<sup>(i)</sup> + b)** = **z<sup>(i)</sup>** 

   `they are the i-th example`

   **Loss(error) function** 
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+L%28%5Chat+y%2C+y%29+%3D+%5Cfrac%7B%28%5Chat+y-+y%29%5E2+%7D%7B2%7D+" 
alt="L(\hat y, y) = \frac{(\hat y- y)^2 }{2} ">

   ` Loss function is defined so as to see how good the output ŷ compared to output label y ` 

   **Logistic Regression Loss Function**

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+L%28%5Chat+y%2C+y%29+%3D+-%28ylog+%5Chat+y+%2B+%281-y%29log%281-%5Chat+y%29%29" 
alt="L(\hat y, y) = -(ylog \hat y + (1-y)log(1-\hat y))">

   If y = 1 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+L%28%5Chat+y%2C+y%29+%3D+-%28log+%5Chat+y%29+" 
alt="L(\hat y, y) = -(log \hat y) ">

   If y = 0
   <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+L%28%5Chat+y%2C+y%29+%3D+-%28log%281-%5Chat+y%29%29+" 
alt="L(\hat y, y) = -(log(1-\hat y)) ">
   Single Training : **COST FUNCTION** 

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+J%28w%2Cb%29+%3D+%5Cfrac%7B1%7D%7Bm%7D++%5Csum_%7Bi%3D1%7D%5Em+L%28%5Chat+y%5E%7B%28i%29%7D+%2C+y%5E%7B%28i%29%7D%29+%3D+-%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+y%5E%7B%28i%29%7Dlog%28%5Chat+y%5E%7B%28i%29%7D%29%2B%281-y%5E%7B%28i%29%7Dlog%281-%5Chat+y%5E%7B%28i%29%7D%29%29" 
alt="J(w,b) = \frac{1}{m}  \sum_{i=1}^m L(\hat y^{(i)} , y^{(i)}) = -\frac{1}{m} \sum_{i=1}^m y^{(i)}log(\hat y^{(i)})+(1-y^{(i)}log(1-\hat y^{(i)}))">

   ---
   ### Gradient Descent

   Cost Function measures how good the parameters w and b are doing on a learning set

   Now, to find w, b that minimizes J(w,b)

   ![Gradient](https://i.ibb.co/6cQ74nw/gradient.png)

   Here, J(w,b) is convex

   Now, we have to initialize, w and b as some value, usually **w = b = 0**

   and after that the gradient starts taking a descent as quickly as possible
   After each iteration, it goes down and down, and hence reaches the global optimum

   ![Gradient](https://i.ibb.co/myY9mxH/gradient2.png)

   ##### Gradient Descent only taking w 

   for, a function, J(w)

   ![Gradient](https://i.ibb.co/hFW6v4F/gradient3.png)

   Repeat, 
         {
               <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+w+%3A%3D+w+-+%5Calpha+%28%5Cfrac%7BdJ%28w%29%7D%7Bdw%7D%29" 
alt="w := w - \alpha (\frac{dJ(w)}{dw})">
         }
               <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+w+%3A%3D+w+-+%5Calpha+%28dw%29" 
alt="w := w - \alpha (dw)">
            
   `where α is learning rate`

   Now, taking two examples

   ![Gradient](https://i.ibb.co/9mXNQ0x/gradient4.png)

   1. When w is taken _too large_, the value of w slowly starts decreasing per the J(w) graph.
      Here 
      <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cfrac%7BdJ%28w%29%7D%7Bdw%7D+%3E+0+" 
alt="\frac{dJ(w)}{dw} > 0 ">
   2. When w is taken _too small_, the value of w slowly starts increasing per the J(w) graph.
      Here 
      <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cfrac%7BdJ%28w%29%7D%7Bdw%7D+%3C+0+" 
alt="\frac{dJ(w)}{dw} < 0 ">

   ##### Gradient Descent taking w and b both

   Repeat,
         {
               <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+w+%3A%3D+w+-%5Calpha+%28%5Cfrac%7B%5Cpartial+J%28w%2Cb%29%7D%7Bdw%7D%29" 
alt="w := w -\alpha (\frac{\partial J(w,b)}{dw})">
         }
         {  
               <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++b+%3A%3D+b+-+%5Calpha+%28%5Cfrac%7B%5Cpartial+J%28w%2Cb%29%7D%7Bdw%7D%29" 
alt=" b := b - \alpha (\frac{\partial J(w,b)}{dw})">
         }

   ### Derivatives

   Slope of function = **<span class="frac"><sup>dy</sup><span>/</span><sub>dx</sub></span>**
   ` for a straight line, derivative does not change`


   ### Computational Graphs

   Computations of a neural network are organized in terms of a forward pass or a forward propagation step, in which we compute the output of the neural network, followed by a backward pass or back propagation step, which we use to compute gradients or compute derivatives

   **For Example:-**
   Take a function  **J(a,b,c) = 3 (a + bc)** 

   First, we compute the value of *bc* , assigning a variable *u* to it

**u = bc** 

   now, we go ahead and calculate **a + bc = a + u**

   we will assign this a variable _v_

   we get **a + u = v**

   and finally, to get the value of the function J, we get 
   **J = 3v**

   ![compgraph](https://i.ibb.co/RzCzvwS/compgraph.png)


   The computation graph organizes a computation with this  <span style="color:blue">*blue* </span> arrow, left-to-right computation and  <span style="color:red"> *red* </span> arrow, right-to-left computation.

   ### Derivatives  by Computation Graph

   In this, we use the method of back-propogation, denoted by the <span style="color:red"> *red* </span> arrows

   we take the previous example, **J(a,b,c) = 3(a+bc)**

   but instead of going from variables a,b,c towards final solution J, we instead work backwards using the various variables we have created to simplify the computation
   ![compgraphder](https://i.ibb.co/JCw6Z7G/compgrapheg1.png)
   The above calculation we see is for calculating a variable, ***d<sub>a*** , which is actually  **<span class="frac"><sup>dJ</sup><span>/</span><sub>dA</sub></span>**

   where J is the final output variable.
   We apply the *chain rule* in this scenario, where 

   ![compgraphder](https://i.ibb.co/94Nr83F/compgrapheg2.png)

   similarly, we calculate ***d<sub>b*** and ***d<sub>c*** which is actually   **<span class="frac"><sup>dJ</sup><span>/</span><sub>db</sub></span>**  and   **<span class="frac"><sup>dJ</sup><span>/</span><sub>dc</sub></span>**

   we can also go ahead and generalise it as finding a new variable, ***d<sub>var***, which is given by

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cfrac%7Bd%28finalOutputVar%29%7D%7Bd%28Var%29%7D" 
alt="\frac{d(finalOutputVar)}{d(Var)}">

   
   ### Logistic Regression Gradient Descent

   We know that, 
   **Logistic Regression Loss Function**
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L%28%5Chat+y%2C+y%29+%3D+-%28ylog+%5Chat+y%2C+%2B+%281-y%29log%281-%5Chat+y%2C%29%29+++%5C%2C++and+%5C%3B+also%2C+%5Chat+y+%3D+a+%3D+%5Csigma%28z%29" >


to get

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+L%28a%2C+y%29+%3D+-%28yloga+%2B+%281-y%29log%281-a%29%29" 
alt="L(a, y) = -(yloga + (1-y)log(1-a))">

   **Linear Regression Function**
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+z+%3D+w%5ET%28x%29+%2B+b+" 
alt="z = w^T(x) + b ">

   to calculate **z**

   say, we have two features, *x<sub>1</sub>* and *x<sub>2</sub>*, we input w<sub>1</sub>* , w<sub>2</sub>* and b

   so, we get

   **z = w<sub>1</sub> x<sub>1</sub> + w<sub>2</sub> x<sub>2</sub> + b**

   then we compute,  **ŷ = a = σ(z)** and then we compute  **L(a,y)**


   from, **dw<sub>1</sub><sup>(i)</sup>  ;  dw<sub>2</sub><sup>(i)</sup>  ;  db<sup>(i)</sup>**


now, computing it backwards first we do 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cfrac%7BdL%28a%2Cy%29%7D%7Bda%7D++%3D+d_a+%3D+-%5Cfrac%7By%7D%7Ba%7D+%2B++++%5Cfrac%7B1-y%7D%7B1-a%7D+%0A++" 
alt="\frac{dL(a,y)}{da}  = d_a = -\frac{y}{a} +    \frac{1-y}{1-a} ">

   from, z = w<sub>1</sub> x<sub>1</sub> + w<sub>2</sub> x<sub>2</sub> + b 

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+d_z+%3D+%5Cfrac%7BdL%7D%7Bdz%7D+%3D+%5Cfrac%7BdL%28a%2Cy%29%7D%7Bdz%7D+" 
alt="d_z = \frac{dL}{dz} = \frac{dL(a,y)}{dz} ">
   
   simplified to get d<sub>z</sub> = a - y  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+d_z+%3D+%5Cfrac%7BdL%7D%7Bda%7D+%5Cfrac%7Bda%7D%7Bdz%7D" 
alt="d_z = \frac{dL}{da} \frac{da}{dz}">


   Finally, to get w1 w2 and b
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7BdL%7D%7Bdw_1%7D+%3D+dw_1+%3D+x_1+dz+%0A" 
alt="\frac{dL}{dw_1} = dw_1 = x_1 dz 
">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw_2+%3D+x_2+dz+%3B++++%5C%3B+and+%5C%3B++%3B+db+%3D+dz+%0A" 
alt="dw_2 = x_2 dz ;    \; and \;  ; db = dz 
">

 therefore, we get 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+w_1+%3A%3D+w_1+-+%5Calpha+dw_1+%5C%3B+and+%5C%3B++w_2+%3A%3D+w_2+-+%5Calpha+dw_2+%5C%3B++and++%5C%3B+b+%3A%3D+b+%2B+%5Calpha+db+" 
alt="w_1 := w_1 - \alpha dw_1 \; and \;  w_2 := w_2 - \alpha dw_2 \;  and  \; b := b + \alpha db ">

   ### Gradient Descent on m training examples

   >Cost Function 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J%28w%2Cb%29+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+L%28a%5E%28i%29+%2C+y%29+" 
alt="J(w,b) = \frac{1}{m} \sum_{i=1}^m L(a^(i) , y) ">

   also, 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%28i%29%7D+%3D+%5Chat+y%5E%7B%28i%29%7D%3D+%5Csigma+%28z%5E%7B%28i%29%7D%29+%3D+%5Csigma+%28w%5ET+x%5Ei+%2B+b%29+" 
alt="a^{(i)} = \hat y^{(i)}= \sigma (z^{(i)}) = \sigma (w^T x^i + b) ">

   for one training example, taking **dw<sub>1</sub><sup>(i)</sup>  ;  dw<sub>2</sub><sup>(i)</sup>  ;  db<sup>(i)</sup>**

   >now, taking derivative for Cost function, 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial+w_1%7DJ%28w%2Cb%29+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial+w_i%7D+L%28a%5E%7B%28i%29%7D%2C+y%5E%7B%28i%29%7D%29" 
alt="\frac{\partial}{\partial w_1}J(w,b) = \frac{1}{m} \sum_{i=1}^m \frac{\partial}{\partial w_i} L(a^{(i)}, y^{(i)})">

   where
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial+w_i%7D+L%28a%5E%7B%28i%29%7D%2C+y%5E%7B%28i%29%7D%29+%3D++%28dw_1%29%5E%7B%28i%29%7D+-+%28x%5E%7B%28i%29%7D%2C+y%5E%7B%28i%29%7D%29" 
alt="\frac{\partial}{\partial w_i} L(a^{(i)}, y^{(i)}) =  (dw_1)^{(i)} - (x^{(i)}, y^{(i)})">

   This gives us the overall gradient that allows us to use `gradient descent`

   Wrapping all of this up in a concrete algorithm
   ___
   Initialize 
    ***J = 0  ; dw<sub>1</sub> = 0  ;  dw<sub>2</sub> = 0  ;  db = 0*** 

   now, applying a for loop
   for, i = 1 to m 
   >First for loop should be over training examples
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++z%5E%7B%28i%29%7D+%3D+w%5ET+x%5E%7B%28i%29%7D+%2B+b+" 
alt=" z^{(i)} = w^T x^{(i)} + b ">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%28i%29%7D+%3D+%5Csigma+%28z%5E%7B%28i%29%7D%29++" 
alt="a^{(i)} = \sigma (z^{(i)})  ">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J+%2B+%3D+y%5E%7B%28i%29%7Dloga%5E%7B%28i%29%7D+%2B+%281-y%5E%7B%28i%29%7D%29log%281-a%5E%7B%28i%29%7D%29" 
alt="J + = y^{(i)}loga^{(i)} + (1-y^{(i)})log(1-a^{(i)})">

   >Second for loop should be over these features
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dz%5E%7B%28i%29%7D+%3D+a%5E%7B%28i%29%7D+-+y%5E%7B%28i%29%7D" 
alt="dz^{(i)} = a^{(i)} - y^{(i)}">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw_1+%2B+%3D+x_1%5E%7B%28i%29%7D+dz%5E%7B%28i%29%7D" 
alt="dw_1 + = x_1^{(i)} dz^{(i)}">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw_2+%2B+%3D+x_2%5E%7B%28i%29%7D+dz%5E%7B%28i%29%7D+" 
alt="dw_2 + = x_2^{(i)} dz^{(i)} ">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+db+%2B+%3D+dz%5E%7B%28i%29%7D" 
alt="db + = dz^{(i)}">

   then finally, for all training examples m, for taking averages
   we get <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J+%2F%3D+m++%3B+%5C%3B++dw_1+%2F%3D+m++%3B+%5C%3B+dw_2+%2F%3Dm++%3B+%5C%3B+db+%2F%3Dm+" 
alt="J /= m  ; \;  dw_1 /= m  ; \; dw_2 /=m  ; \; db /=m ">
   ___

   to implement one stage of gradient descent, we now
   > w1 gets updated as
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++w_1+%3A%3D+w_1+-+%5Calpha+dw_1+" 
alt=" w_1 := w_1 - \alpha dw_1 ">
   >w2 gets updated as
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+w_2+%3A%3D+w_2+-+%5Calpha+dw_2" 
alt="w_2 := w_2 - \alpha dw_2">
   >w3 gets updated as
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+b+%3A%3D+b+-+%5Calpha+db+" 
alt="b := b - \alpha db ">

   ### Vectorization

   Vectorization is applied in order to get rid of explicit for loops 
   for linear regression
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+z+%3D+w%5ET%28x%29+%2B+b+" 
alt="z = w^T(x) + b ">

   w, x are two column vectors where **x** ∈ R<sup>n<sub>x</sub></sup> and **w** ∈ R<sup>n<sub>x</sub></sup>

   for a non-vectorized loop,

   ```
   z = 0
   for i in range(n-x) :
      z += w[i] * x[i]
   z += b
   ```

   for a vectorized loop, using NumPy

   ```
   z = np.dot(w,x) + b
   ```

   now, applying in code

   ```
   import numpy as np

   a = np.array([1,2,3,4])
   print(a)
   ```

   ```
   import time

   a = np.random(1000000)
   b = np.random(1000000)

   tic = time.time()
   c = np.dot (a , b)

   toc = time.time()
   print(c)
   print("Vectorized version" + str (1000*(toc-tic)) + "ms")

   c = 0
   tic = time.time()
   for i in range (1000000) :
      c += a[i] + b[i]
   toc = time.time()

   print("For loop" + str (1000*(toc-tic)) + "ms")
   ```

   > when we output this code, we clearly see the time for the **for loop** is far much more than the **vectorized loop** 

   >SIMD Instructions stand for Single Instruction Multiple Data, it is seen in CPU/GPU paralellelization
   It enables Python NumPy to take advantage of paralellelism, hence reducing the time for processing calculations

   ##### Neural Network Programming Guidelines

   let  u = Av 
         <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++u_i+%3D+%5Csum_%7Bi%3Dj%7D+A_%7Bij%7D+V_j+" 
alt=" u_i = \sum_{i=j} A_{ij} V_j ">

   Non Vectorized 
   ```
   u = np.zeros(n,1)

   for i ...
      for j ...
         u[i] = A[i][j] * V[j]
   ```

   Vectorized
   ```
   u = np.dot(A,v)
   ```

   ###### Vectors and Matrix Valued Functions

   To apply the exponential operation on every element of matrix/vector

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+v+%3D+%5Cbegin%7Bbmatrix%7D+v_1+%5C%5C+v_2+%5C%5C+.+%5C%5C+.+%5C%5C+.+%5C%5C+v_n+%5Cend%7Bbmatrix%7D+" 
alt="v = \begin{bmatrix} v_1 \\ v_2 \\ . \\ . \\ . \\ v_n \end{bmatrix} ">

   so , applying operation

   
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+u+%3D+%5Cbegin%7Bbmatrix%7D+e%5E%7B%28v_1%29%7D+%5C%5C+e%5E%7B%28v_2%29%7D++%5C%5C+.+%5C%5C+.+%5C%5C+.+%5C%5C+e%5E%7B%28v_n%29%7D+%5Cend%7Bbmatrix%7D+" 
alt="u = \begin{bmatrix} e^{(v_1)} \\ e^{(v_2)}  \\ . \\ . \\ . \\ e^{(v_n)} \end{bmatrix} ">

   Non-Vectorized
   ```
   u = np.zeros((n,1))
   for i in range(n) :
      u[i] = math.exp(v[i])
   ```

   Vectorized

   ```
   import numpy as np
   u = np.exp(v)

   print(u)
   ```

   **Now, for gradient descent**

   >  J = 0, db=0 
   > ```
   > dw = np.zeros((n-x, 1))
   > ```
   > for, i = 1 to m 
   >First for loop should be over training examples
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+z%5E%7B%28i%29%7D+%3D+w%5ET+x%5E%7B%28i%29%7D+%2B+b" 
alt="z^{(i)} = w^T x^{(i)} + b">
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%28i%29+%3D+%5Csigma+%28z%5E%7B%28i%29%7D%29++" 
alt="a^(i) = \sigma (z^{(i)})  ">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J+%2B+%3D+y%5E%7B%28i%29%7D+loga%5E%7B%28i%29%7D+%2B+%281-y%5E%7B%28i%29%29%7Dlog%281-a%5E%7B%28i%29%7D%29" 
alt="J + = y^{(i)} loga^{(i)} + (1-y^{(i))}log(1-a^{(i)})">

   >also,
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dz%5E%7B%28i%29+%7D%3D+a%5E%7B%28i%29%7D+-+y%5E%7B%28i%29%7D" alt="dz^{(i) }= a^{(i)} - y^{(i)}">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw+%2B%3D+x%5E%7B%28i%29%7D+dz%5E%7B%28i%29%7D" 
alt="dw += x^{(i)} dz^{(i)}">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+db+%2B%3D+dz%5E%7B%28i%29%7D+" 
alt="db += dz^{(i)} ">

   >then finally, for all training examples m, for taking averages
   we get <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J+%3D+J%2F+m++%3B+%5C%3B+dw+%3D+dw%2Fm++%3B+%5C%3B+db%3D+db%2Fm+" 
alt="J = J/ m  ; \; dw = dw/m  ; \; db= db/m ">

   ### Vectorizing Logical Regression

   Assuming a forward propogation step, in Logistic Regeression, 

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++z%5E%7B%281%29%7D+%3D+w%5ET+x%5E%7B%281%29%7D+%2B+b+" 
alt=" z^{(1)} = w^T x^{(1)} + b "> 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a%5E%7B%281%29%7D+%3D%5Csigma+%28z%5E%7B%281%29%7D%29+" 
alt="a^{(1)} =\sigma (z^{(1)}) ">

   similarly, 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+z%5E%7B%282%29%7D++%3D+w%5ET+x%5E%7B%282%29%7D+%2B+b+" 
alt="z^{(2)}  = w^T x^{(2)} + b ">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a%5E%7B%282%29%7D+%3D%5Csigma+%28z%5E%7B%282%7D%29+" 
alt="a^{(2)} =\sigma (z^{(2}) ">

   and so on and so forth...

   we have defined 

   <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+X+%3D%5Cbegin%7Bbmatrix%7D+%7C%26+%7C%26+%7C%26+%7C%26+%7C%26+%7C+%5C%5C%5C+x%5E%281%29+%26+x%5E%282%29+%26+.+%26+.+%26+.+%26+x%5E%28m%29+%5C%5C%5C+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C%5Cend%7Bbmatrix%7D" 
alt="X =\begin{bmatrix} |& |& |& |& |& | \\\ x^(1) & x^(2) & . & . & . & x^(m) \\\ | & | & | & | & | & |\end{bmatrix}">
   it is a (n<sub>x</sub> , m) vector


   Similarly, for finding Z

   we construct another Row Vector

   <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+Z+%3D++%5Cbegin%7Bbmatrix%7D%5C+z%5E1+%26+z%5E2+%26+.%26+.%26.+%26+z%5Em%5Cend%7Bbmatrix%7D+%3D+w%5ET+X+%2B+%5Cbegin%7Bbmatrix%7D%5C+b+%26+b+%26+.+%26+.+%26.+%26+b+%5Cend%7Bbmatrix%7D+" 
alt="Z =  \begin{bmatrix}\ z^1 & z^2 & .& .&. & z^m\end{bmatrix} = w^T X + \begin{bmatrix}\ b & b & . & . &. & b \end{bmatrix} ">

   where both the z vector and b vectors are of dimension **(1 x m)**

   now, 

   we have, <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle++w%5ET+X+%3D+w%5ET+%5Cbegin%7Bbmatrix%7D+%7C%26+%7C%26+%7C%26+%7C%26+%7C%26+%7C+%5C%5C%5C+x%5E%281%29+%26+x%5E%282%29+%26+.+%26+.+%26+.+%26+x%5E%28m%29+%5C%5C%5C+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C%5Cend%7Bbmatrix%7D+" 
alt=" w^T X = w^T \begin{bmatrix} |& |& |& |& |& | \\\ x^(1) & x^(2) & . & . & . & x^(m) \\\ | & | & | & | & | & |\end{bmatrix} ">

   calculating further, we get

   <img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+Z+%3D+w%5ET+X+%2B+b++%3D+%5Cbegin%7Bbmatrix%7D+%7C%26+%7C%26+%7C%26+%7C%26+%7C%26+%7C+%5C%5C%5C+w%5ET+x%5E%281%29+%2B+b+%26+w%5ET+x%5E%282%29+%2B+b++%26+.+%26+.+%26+.+%26+w%5ET+x%5E%28m%29+%2B+b+%5C%5C%5C+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C%5Cend%7Bbmatrix%7D" 
alt="Z = w^T X + b  = \begin{bmatrix} |& |& |& |& |& | \\\ w^T x^(1) + b & w^T x^(2) + b  & . & . & . & w^T x^(m) + b \\\ | & | & | & | & | & |\end{bmatrix}">

   to implement **Z**

   ```
   Z = np.dot(w.T , X) + b 
   ```

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++b+%5Cepsilon+%5CBbb+R++" 
alt=" b \epsilon \Bbb R  "> but when added with a vector, python automatically transforms it into <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Cbegin%7Bbmatrix%7D%5C+b+%26+b+%26+.+%26+.+%26.+%26+b+%5Cend%7Bbmatrix%7D+" 
alt="\begin{bmatrix}\ b & b & . & . &. & b \end{bmatrix} ">

   This is called ***broadcasting***

   for calulating <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+A+%3D+%5Cbegin%7Bbmatrix%7D%5C+a%5E1+%26+a%5E2+%26+.+%26+.+%26.+%26+a%5Em+%5Cend%7Bbmatrix%7D+%3D+%5Csigma+%28Z%29" 
alt="A = \begin{bmatrix}\ a^1 & a^2 & . & . &. & a^m \end{bmatrix} = \sigma (Z)">

   ### Vectorizing Logistic Regression

   for gradient computation,
   we computed, <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dz%5E1%3D+a%5E1+-+y%5E1" 
alt="dz^1= a^1 - y^1"> and <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+dz%5E2+%3D+a%5E2+-+y%5E2+" 
alt="dz^2 = a^2 - y^2 "> up until **m** examples

  <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+dZ+%3D+%5Cbegin%7Bbmatrix%7D+dz%5E1+%26+dz%5E2+%26+.+%26+.+%26+.+%26+dz%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt="dZ = \begin{bmatrix} dz^1 & dz^2 & . & . & . & dz^m \\\end{bmatrix} ">

   also for activation function,

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+A+%3D+%5Cbegin%7Bbmatrix%7D+a%5E1+%26+a%5E2+%26+.+%26+.+%26+.+%26+a%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt="A = \begin{bmatrix} a^1 & a^2 & . & . & . & a^m \\\end{bmatrix} ">

   also <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+Y+%3D+%5Cbegin%7Bbmatrix%7D+y%5E1+%26+y%5E2+%26+.+%26+.+%26+.+%26+y%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt="Y = \begin{bmatrix} y^1 & y^2 & . & . & . & y^m \\\end{bmatrix} ">

   we can also calculate further

 <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+dZ+%3D+A+-+Y+%3D+%5Cbegin%7Bbmatrix%7D+a%5E1+-+y%5E1+%26+a%5E2+-+y%5E2+%26+.+%26+.+%26+.+%26+a%5Em+-+y%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt="dZ = A - Y = \begin{bmatrix} a^1 - y^1 & a^2 - y^2 & . & . & . & a^m - y^m \\\end{bmatrix} ">

   with one line of code, we can compute  dZ 


   ```
   dw = 0

   dw += x1 dz1
   dw += x2 dz2
   .
   .
   dw /= m

   ```
   we still have for loop over these

   and similarly for db

   ```
   db = 0 

   db += dz1
   db += dz2
   .
   .
   db += dzm

   db /=m
   ```

   so basically, <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+db+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+dz%5Ei+" 
alt="db = \frac{1}{m} \sum_{i=1}^m dz^i ">

   for python then 

   ```
   db = 1/m np.sum(dz) 
   ```

   now for, <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+dw+%3D+%5Cfrac%7B1%7D%7Bm%7D+X+dZ%5ET+%0A+++" 
alt="dw = \frac{1}{m} X dZ^T 
   ">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Cbegin%7Bbmatrix%7D+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C+%5C%5C+x%5E1+%26+.+%26+.+%26+.+%26+x%5Em+%5C%5C+%7C+%26+%7C+%26+%7C+%26+%7C+%26+%7C+%5Cend%7Bbmatrix%7D+%5Cbegin%7Bbmatrix%7D+dz%5E1+%5C%5C+.+%5C%5C+.+%5C%5C+.+%5C%5C+dz%5Em+%5Cend%7Bbmatrix%7D" 
alt="dw = \frac{1}{m} \begin{bmatrix} | & | & | & | & | \\ x^1 & . & . & . & x^m \\ | & | & | & | & | \end{bmatrix} \begin{bmatrix} dz^1 \\ . \\ . \\ . \\ dz^m \end{bmatrix}">

   equating to

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+dw+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Cbegin%7Bbmatrix%7D+x%5E1+dz%5E1+%2B+%26+.%26+.+%26.+%26+%2B+x%5Em+dz%5Em+%5Cend%7Bbmatrix%7D+" 
alt="dw = \frac{1}{m} \begin{bmatrix} x^1 dz^1 + & .& . &. & + x^m dz^m \end{bmatrix} ">

   **basically, the sequence of events**

   >we take A and Y, where, <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++A+%3D+%5Cbegin%7Bbmatrix%7D+a%5E1+%26+a%5E2+%26+.+%26+.+%26+.+%26+a%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt=" A = \begin{bmatrix} a^1 & a^2 & . & . & . & a^m \\\end{bmatrix} ">
   >
   >and <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+Y+%3D+%5Cbegin%7Bbmatrix%7D+y%5E1+%26+y%5E2+%26+.+%26+.+%26+.+%26+y%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt="Y = \begin{bmatrix} y^1 & y^2 & . & . & . & y^m \\\end{bmatrix} ">
   >then performing, 
   >Z = A - Y
   > then,
   ><img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dZ+%3D+A+-+Y+%3D+%5Cbegin%7Bbmatrix%7D+a%5E1+-+y%5E1+%26+a%5E2+-+y%5E2+%26+.+%26+.+%26+.+%26+a%5Em+-+y%5Em+%5C%5C%5Cend%7Bbmatrix%7D+" 
alt="dZ = A - Y = \begin{bmatrix} a^1 - y^1 & a^2 - y^2 & . & . & . & a^m - y^m \\\end{bmatrix} ">
   > then
   > ` db = 1/m np.sum(dz) `
   > and finally
   > <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw+%3D+%5Cfrac%7B1%7D%7Bm%7D+X+dZ%5ET+" 
alt="dw = \frac{1}{m} X dZ^T ">
   > to get  **dw, db , dZ** 


   #### Implementing Logistic Regression Vectorized
   for iteration in range(1000) :
   {
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++Z+%3D+w%5ET+X+%2B+b+" 
alt=" Z = w^T X + b ">
   ` z = np.dot(w.T,x)+b`
   for A
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+A+%3D+%5Csigma+%28Z%29+" 
alt="A = \sigma (Z) ">

   for Z
    dZ = A- Y 

   for w
    <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+dw+%3D+%5Cfrac%7B1%7D%7Bm%7D+X+dZ%5ET+" 
alt="dw = \frac{1}{m} X dZ^T ">

   for b
   ` db = 1/m np.sum(dZ) `

   ###### We have done both forward and back propogation

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+w+%3A%3D+w+-+%5Calpha+dw" 
alt="w := w - \alpha dw">
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+b+%3A%3D+b+-+%5Calpha+db" 
alt="b := b - \alpha db">
   }

   ###### This involves a single iteration of gradient descent, which is then looped 


   ### Broadcasting in Python

   Example for Broadcasting

   Example matrix A

   <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++A+%3D+%5Cbegin%7Bbmatrix%7D+56.0+%26++0.0+%26+4.4+%26+68.0+%5C%5C+1.2+%26+104.0+%26+52.0+%26+9.0+%5C%5C+1.8+%26+135.0+%26+99.0+%26+0.9+%5Cend%7Bbmatrix%7D+" 
alt=" A = \begin{bmatrix} 56.0 &  0.0 & 4.4 & 68.0 \\ 1.2 & 104.0 & 52.0 & 9.0 \\ 1.8 & 135.0 & 99.0 & 0.9 \end{bmatrix} ">

   to calculate average per column without explicit for loop

   ##### As a code.

   ```
   import numpy as np

   A = np.array([[56.0 , 0.0 , 4.4 , 68.0], 
               [1.2 , 104.0 , 52.0 ,9.0],
               [1.8 , 135.0 , 99.0 ,0.9]])

   print(A)

   cal = A.sum(axis = 0) #this is a (1,4) matrix

   print(cal)

   percentage = 100*A/cal.reshape(1,4)  #here the .reshape is redundant as cal is a (1,4) matrix already

   print(percentage)
   ```


   ###### GENERAL PRINCIPLE 

   if you have a (m , n) matrix, {any operation} , (1 , n) matrix ---> Transformed into a (m , n) matrix


   #### Python-Numpy Vectors
   ```
   import numpy as np

   a = np.random.randn(5)  #creates 5 random gaussian vars

   print(a)

   print(a.shape)
   #output = (5, ) this is a rank one array (not a row/neither a column)

   print(a.T)
   #output same as a

   print(np.dot(a, a.T))
   ```
   **DONT USE RANK 1 ARRAYS**
   ```
   b = np.random.randn(5,1)

   print(b) #gives a column vector

   print(b.T)
   #output gives a row vector

   print(np.dot(a, a.T))
   #output gives a (5,5) vector

   ```

   (Week 3)
---
## Shallow Neural Networks
---
### What is a Neural Network
![NN](https://i.ibb.co/7VXjKdV/neuralnw.png)

So, for a neural network, they are multiple stacked sigmoid function units taking data from a previous set, passing it to the next. Hence, taking a simple neural network example, taking input layer x<sub>1</sub> x<sub>2</sub> x<sub>3</sub> 
Taking layers  **[1] [2] \; and\; [3]**  
for each layer, we get a different sigmoid function

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle++z%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7D+x+%2B+b%5E%7B%5B1%5D%7D+%3D%3E+a%5E%7B%5B1%5D%7D+%3D+%5Csigma+%28z%5E%7B%5B1%5D%7D%29+%3D%3E+%5Cmathcal+L+%28a%5E%7B%5B1%5D%7D%2C+y%29+" 
alt=" z^{[1]} = W^{[1]} x + b^{[1]} => a^{[1]} = \sigma (z^{[1]}) => \mathcal L (a^{[1]}, y) ">

which then passes it on to layer  **[2]**

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7D+x+%2B+b%5E%7B%5B2%5D%7D+%3D%3E+a%5E%7B%5B2%5D%7D+%3D+%5Csigma+%28z%5E%7B%5B2%5D%7D%29+%3D%3E+%5Cmathcal+L+%28a%5E%7B%5B2%5D%7D%2C+y%29+" 
alt="z^{[2]} = W^{[2]} x + b^{[2]} => a^{[2]} = \sigma (z^{[2]}) => \mathcal L (a^{[2]}, y) ">

and so on and so forth

Basically, it is taking a logistic regression, and then repeat it 'n' times, here, two times


### Neural Network Representation and Computation

for a 2 layered NN
they will have, input layer, hidden layer and an output layer

in a generalised way,

![NN](https://i.ibb.co/4T6WbRb/2layerNN.png)

Input layer is <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B0%5D%7D_%7Bn%7D+" 
alt="a^{[0]}_{n} "> , where n is the nth single neuron unit (Logistic Regression Unit) 

Hidden layer is <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B1%5D%7D_%7Bn%7D+" 
alt="a^{[1]}_{n} "> , where n is the nth single neuron unit (Logistic Regression
Unit)
the hidden layer is given by a column matrix <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a%5E%7B%5B1%5D%7D+%3D+%5Cbegin%7Bbmatrix%7D+a%5E%7B%5B1%5D%7D_%7B1%7D+%5C%5C+a%5E%7B%5B1%5D%7D_%7B2%7D+%5C%5Ca%5E%7B%5B1%5D%7D_%7B3%7D+%5C%5C+a%5E%7B%5B1%5D%7D_%7B4%7D+%5Cend%7Bbmatrix%7D+" 
alt="a^{[1]} = \begin{bmatrix} a^{[1]}_{1} \\ a^{[1]}_{2} \\a^{[1]}_{3} \\ a^{[1]}_{4} \end{bmatrix} ">

and Output layer is **a<sup>[2] </sup>** 

![layer](https://i.ibb.co/JnhbNwJ/2part2layer.png)
from this, we see each logistic regression unit divided into two parts, one being calculation of  z = w<sup>T</sup> x + b  and other being the calculation of σ(z) 

so similarly taking this calculations for every logistical regression, we see that the entire neuron calculates

for each layer,

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+z%5E%7B%5B1%5D%7D_%7B1%7D+%3D+++w%5E%7B%5B1%5DT%7D_%7B1%7Dx+%2B+b%5E%7B%5B1%5D%7D_%7B1%7D+%5C%3B++and+%5C%3B++a%5E%7B%5B0%5D%7D_%7B1%7D+%3D+%5Csigma%28z%5E%7B%5B1%5D%7D%7B1%7D%29" 
alt="z^{[1]}_{1} =   w^{[1]T}_{1}x + b^{[1]}_{1} \;  and \;  a^{[0]}_{1} = \sigma(z^{[1]}{1})">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+z%5E%7B%5B1%5D%7D_%7B2%7D+%3D+++w%5E%7B%5B1%5DT%7D_%7B2%7Dx+%2B+b%5E%7B%5B1%5D%7D_%7B2%7D+%5C%3B+and+%5C%3B++a%5E%7B%5B0%5D%7D_%7B2%7D+%3D+%5Csigma%28z%5E%7B%5B1%5D%7D%7B2%7D%29" 
alt="z^{[1]}_{2} =   w^{[1]T}_{2}x + b^{[1]}_{2} \; and \;  a^{[0]}_{2} = \sigma(z^{[1]}{2})">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+z%5E%7B%5B1%5D%7D_%7B3%7D%3D+++w%5E%7B%5B1%5DT%7D_%7B3%7Dx+%2B+b%5E%7B%5B1%5D%7D_%7B3%7D+%5C%3B+and+%5C%3B++a%5E%7B%5B0%5D%7D_%7B3%7D+%3D+%5Csigma%28z%5E%7B%5B1%5D%7D%7B3%7D%29" 
alt="z^{[1]}_{3}=   w^{[1]T}_{3}x + b^{[1]}_{3} \; and \;  a^{[0]}_{3} = \sigma(z^{[1]}{3})">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++z%5E%7B%5B1%5D%7D_%7B4%7D+%3D+++w%5E%7B%5B1%5DT%7D_%7B4%7Dx+%2B+b%5E%7B%5B1%5D%7D_%7B4%7D+%5C%3B+and+%5C%3B++a%5E%7B%5B0%5D%7D_%7B4%7D+%3D+%5Csigma%28z%5E%7B%5B1%5D%7D_%7B4%7D%29+" 
alt=" z^{[1]}_{4} =   w^{[1]T}_{4}x + b^{[1]}_{4} \; and \;  a^{[0]}_{4} = \sigma(z^{[1]}_{4}) ">

hence, creating a column vector for each

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+z%5E%7B%5B1%5D%7D+%3D+%5Cbegin%7Bbmatrix%7D%7B-%7D+%26%7Bw%5E%7B%5B1%5DT%7D_%7B1%7D%7D++%26+%7B-%7D%5C%5C%7B-%7D+%26%7Bw%5E%7B%5B1%5DT%7D_%7B2%7D%7D+%26+%7B-%7D++%5C%5C+%7B-%7D+%26%7Bw%5E%7B%5B1%5DT%7D_%7B3%7D%7D+%26+%7B-%7D+%5C%5C+%7B-%7D+%26%7Bw%5E%7B%5B1%5DT%7D_%7B4%7D%7D++%26+%7B-%7D++%5Cend%7Bbmatrix%7D+%5Cbegin%7Bbmatrix%7D+x_%7B1%7D+%5C%5C+x_%7B2%7D+%5C%5C+x_%7B3%7D+%5Cend%7Bbmatrix%7D+%2B+%5Cbegin%7Bbmatrix%7D+b%5E%7B%5B1%5D%7D_%7B1%7D+%5C%5C+b%5E%7B%5B1%5D%7D_%7B2%7D+%5C%5Cb%5E%7B%5B1%5D%7D_%7B3%7D+%5C%5C+b%5E%7B%5B1%5D%7D_%7B4%7D+%5Cend%7Bbmatrix%7D+%3D+%5Cbegin%7Bbmatrix%7D+%7Bw%5E%7B%5B1%5DT%7D_%7B1%7D+x+%2B+b%5E%7B%5B1%5D%7D_%7B1%7D%7D%5C%5C%7Bw%5E%7B%5B1%5DT%7D_%7B2%7D+x+%2B+b%5E%7B%5B1%5D%7D_%7B2%7D%7D%5C%5C%7Bw%5E%7B%5B1%5DT%7D_%7B3%7D+x+%2B+b%5E%7B%5B1%5D%7D_%7B3%7D%7D%5C%5C%7Bw%5E%7B%5B1%5DT%7D_%7B4%7D+x+%2B+b%5E%7B%5B1%5D%7D_%7B4%7D%7D++%5Cend%7Bbmatrix%7D+%3D+%5Cbegin%7Bbmatrix%7D+z%5E%7B%5B1%5D%7D_%7B1%7D+%5C%5C+z%5E%7B%5B1%5D%7D_%7B2%7D+%5C%5Cz%5E%7B%5B1%5D%7D_%7B3%7D+%5C%5C+z%5E%7B%5B1%5D%7D_%7B4%7D+%5Cend%7Bbmatrix%7D+" 
alt="z^{[1]} = \begin{bmatrix}{-} &{w^{[1]T}_{1}}  & {-}\\{-} &{w^{[1]T}_{2}} & {-}  \\ {-} &{w^{[1]T}_{3}} & {-} \\ {-} &{w^{[1]T}_{4}}  & {-}  \end{bmatrix} \begin{bmatrix} x_{1} \\ x_{2} \\ x_{3} \end{bmatrix} + \begin{bmatrix} b^{[1]}_{1} \\ b^{[1]}_{2} \\b^{[1]}_{3} \\ b^{[1]}_{4} \end{bmatrix} = \begin{bmatrix} {w^{[1]T}_{1} x + b^{[1]}_{1}}\\{w^{[1]T}_{2} x + b^{[1]}_{2}}\\{w^{[1]T}_{3} x + b^{[1]}_{3}}\\{w^{[1]T}_{4} x + b^{[1]}_{4}}  \end{bmatrix} = \begin{bmatrix} z^{[1]}_{1} \\ z^{[1]}_{2} \\z^{[1]}_{3} \\ z^{[1]}_{4} \end{bmatrix} ">

and activation function

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a%5E%7B%5B1%5D%7D+%3D+%5Cbegin%7Bbmatrix%7D+a%5E%7B%5B1%5D%7D_%7B1%7D+%5C%5C+a%5E%7B%5B1%5D%7D_%7B2%7D+%5C%5Ca%5E%7B%5B1%5D%7D_%7B3%7D+%5C%5C+a%5E%7B%5B1%5D%7D_%7B4%7D+%5Cend%7Bbmatrix%7D+%3D+%5Csigma+%28z%5E%7B%5B1%5D%7D%29" 
alt="a^{[1]} = \begin{bmatrix} a^{[1]}_{1} \\ a^{[1]}_{2} \\a^{[1]}_{3} \\ a^{[1]}_{4} \end{bmatrix} = \sigma (z^{[1]})">


so, finally, taking all of this into consideration, 
![rr](https://i.ibb.co/q1g27qt/repre.png)
here we can see the dimensions of each vector involved in a 2Layer Neural Network

### Vectorizing Across Multiple Examples

![layer](https://i.ibb.co/gJGWBS0/multi1.png)
![layer](https://i.ibb.co/0JrhLRn/multi2.png)


here, we can see that the for loop needs to be vectorized for all the **m** training examples
taking a vector for each X,  Z and A

here, the<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AZ%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7DX+%2B+b%5E%7B%5B1%5D%7D+" 
alt="
Z^{[1]} = W^{[1]}X + b^{[1]} ">

solves it for i = 1 to m 
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++z%5E%7B%7B1%7D%28i%29%7D+%3D+w%5E%7B%5B1%5D%7D+x%5E%7B%28i%29%7D+%2B+b%5E%7B%5B1%5D%7D+" 
alt=" z^{{1}(i)} = w^{[1]} x^{(i)} + b^{[1]} ">

similarly <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+A%5E%7B%5B1%5D%7D+%3D+%5Csigma+%28Z%5E%7B%5B1%5D%7D%29+" 
alt="A^{[1]} = \sigma (Z^{[1]}) ">

solves it for i = 1 to m
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B1%5D%28i%29%7D+%3D+%5Csigma+%28z%5E%7B%5B1%5D%7D%29+" 
alt="a^{[1](i)} = \sigma (z^{[1]}) ">

and, <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++Z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7D+X%5E%7B%5B1%5D%7D+%2B+B%5E%7B%5B2%5D%7D+" 
alt=" Z^{[2]} = W^{[2]} X^{[1]} + B^{[2]} ">
solves it for i = 1 to m
   <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+z%5E%7B%7B2%7D%28i%29%7D+%3D+w%5E%7B%5B2%5D%7D+x%5E%7B%28i%29%7D+%2B+b%5E%7B%5B2%5D%7D+" 
alt="z^{{2}(i)} = w^{[2]} x^{(i)} + b^{[2]} ">

also, <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+A%5E%7B%5B2%5D%7D+%3D+%5Csigma+%28Z%5E%7B%5B2%5D%7D%29+" 
alt="A^{[2]} = \sigma (Z^{[2]}) ">

solves it for
  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B2%5D%28i%29%7D+%3D+%5Csigma+%28z%5E%7B%5B2%5D%28i%29%7D%29+" 
alt="a^{[2](i)} = \sigma (z^{[2](i)}) ">

where **X** **Z** and **A** are the vectors as defined above.

### Activation Functions

Sigmoid is a type of activation function

so, taking example of a 2 layered neural network
x1/x2/x3 as inputs and  ŷ output with 3 hidden layers

> Given x : 
> <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AZ%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7DX+%2B+b%5E%7B%5B1%5D%7D+" 
alt="
Z^{[1]} = W^{[1]}X + b^{[1]} ">
>  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AA%5E%7B%5B1%5D%7D+%3D+g%5E%7B%5B1%5D%7D%28Z%5E%7B%5B1%5D%7D%29+" 
alt="
A^{[1]} = g^{[1]}(Z^{[1]}) ">
> 
>  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0A+Z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7DA%5E%7B%5B1%5D%7D+%2B+b%5E%7B%5B2%5D%7D+" 
alt="
 Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} ">
>  <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++a%5E%7B%5B2%5D%7D+%3D++g%28z%5E%7B%5B2%5D%7D%29+" 
alt=" a^{[2]} =  g(z^{[2]}) ">

where ***g*** is a nonlinear activation function

for example

instead of a sigmoid function, we can always use the <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++a+%3D+tanh+%28z%29+%3D+%5Cfrac+%7Be%5E%7Bz%7D+-+e%5E%7B-z%7D%7D%7Be%5E%7Bz%7D+-+e%5E%7B-z%7D%7D" 
alt=" a = tanh (z) = \frac {e^{z} - e^{-z}}{e^{z} - e^{-z}}"> function

>tanh(x) is given by
![tanh](https://mathworld.wolfram.com/images/interactive/TanhReal.gif)

**we can also have different activation functions for each layer**

downsides are if z values are very large, the gradient descent is slowed 

##### Rectified Linear Unit 


>This is the plot of the *Rectified Linear Unit* and the formula is given by <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++a+%3D+max%280%2Cz%29+" 
alt=" a = max(0,z) ">
slope is -ve when z is +ve and slope is 0 when z is -ve 
![relu](https://machinelearningmastery.com/wp-content/uploads/2018/10/Line-Plot-of-Rectified-Linear-Activation-for-Negative-and-Positive-Inputs.png)

**Some rules for choosing activating functions**

If output is zero-one value (binary classification), then the sigmoid activation function is the way to go, but for all other units, the Rectified Linear Unit is the way to go

Disadvantage of ReLU is slope is 0 when z is -ve

***Leaky ReLU***

the Leaky ReLU is more better than the normal ReLU function, as it's slope is not 0 when z is -ve

![leaky](https://1.bp.blogspot.com/-5ymhxBydo8A/XPj_qXK-sWI/AAAAAAAABU4/UjgZ7eChpwsoPa1_bZjvdrzKCsCfQPaJgCLcBGAs/s1600/leaking_relu_2.PNG)

For a lot of space of z the derivative of activation function is not 0, hence the advantage of ReLU and Leaky ReLU.

>Here's the sigmoid activation function. I would say never use this except for the output layer if you're doing binomial classification or maybe almost never use this. And the reason I almost never use this is because the tan h is pretty much strictly superior.
-AndrewNG

![leak](https://i.ibb.co/48jW7wL/activation.png)

### Need for Non-Linear Activation Functions

##### First, we need to talk about Linear Activation functions, 

taking

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%0Aa%5E%7B%5B1%5D%7D+%3D+%5Coperatorname+%7Bg%7D%28z%5E%7B%5B1%5D%7D%29+%3D+z%5E%7B%5B1%5D%7D+" 
alt="
a^{[1]} = \operatorname {g}(z^{[1]}) = z^{[1]} ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%0Ahere%2C+proved+%5C%3Bthat+%5Coperatorname+%7Bg%7D%28z%29+%3D+z+%2C+which+%5C%3B+makes+%5C%2C%5Coperatorname+%7Bg%7D+a+%5C%3Blinear+%5C%3B+function+" 
alt="
here, proved \;that \operatorname {g}(z) = z , which \; makes \,\operatorname {g} a \;linear \; function ">

which gives <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a%5E%7B%5B1%5D%7D+%3D+z%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7Dx+%2B+b%5E%7B%5B1%5D%7D+" 
alt="a^{[1]} = z^{[1]} = W^{[1]}x + b^{[1]} ">

and then <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B2%5D%7D+%3D+z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7D+a%5E%7B%5B1%5D%7D+%2B+b%5E%7B%5B2%5D%7D+" 
alt="a^{[2]} = z^{[2]} = W^{[2]} a^{[1]} + b^{[2]} ">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B2%5D%7D+%3D+z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7D+%28W%5E%7B%5B1%5D%7Dx+%2B+b%5E%7B%5B1%5D%7D%29+%2B+b%5E%7B%5B2%5D%7D+" 
alt="a^{[2]} = z^{[2]} = W^{[2]} (W^{[1]}x + b^{[1]}) + b^{[2]} ">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B2%5D%7D+%3D+z%5E%7B%5B2%5D%7D+%3D+%28W%5E%7B%5B2%5D%7D+W%5E%7B%5B1%5D%7D%29x+%2B+%28W%5E%7B%5B2%5D%7D+b%5E%7B%5B1%5D%7D+%2B+b%5E%7B%5B2%5D%7D%29+" 
alt="a^{[2]} = z^{[2]} = (W^{[2]} W^{[1]})x + (W^{[2]} b^{[1]} + b^{[2]}) ">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B2%5D%7D+%3D+z%5E%7B%5B2%5D%7D+%3D+W%5E%5Cprime+x+%2B+b%5E%5Cprime+%0A" 
alt="a^{[2]} = z^{[2]} = W^\prime x + b^\prime 
">
where, <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++%28W%5E%7B%5B2%5D%7D+W%5E%7B%5B1%5D%7D%29+%3D+W%5E%5Cprime+%5C%3B+and+%5C%3B+%28W%5E%7B%5B2%5D%7D+b%5E%7B%5B1%5D%7D+%2B+b%5E%7B%5B2%5D%7D%29+%3D+b%5E%5Cprime+" 
alt=" (W^{[2]} W^{[1]}) = W^\prime \; and \; (W^{[2]} b^{[1]} + b^{[2]}) = b^\prime ">


When you have a NN with multiple hidden layers and having linear activation function or alternatively, if you don't have an activation function, then no matter how many layers your neural network has, all it's doing is just computing a linear activation function.
A linear hidden layer is more or less useless because the composition of two linear functions is itself a linear function.

### Derivatives of Activation Functions

##### For Sigmoid 

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+g%28z%29+%3D+%5Cfrac%7B1%7D%7B1+%2B+e%5E%7B-z%7D%7D+" 
alt="g(z) = \frac{1}{1 + e^{-z}} ">

and  a = g(z) 
taking derivative

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+g%5E%5Cprime%28z%29+%3D+%5Cfrac+%7Bd%7D%7Bdz%7D+g%28z%29+%3D+g%28z%29%281-+g%28z%29%29+%3D+a%281-a%29" 
alt="g^\prime(z) = \frac {d}{dz} g(z) = g(z)(1- g(z)) = a(1-a)">

again, if z is large, g(z) will be close to 1 and <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+g%5E%5Cprime+%28z%29+%3D+0+" 
alt="g^\prime (z) = 0 ">

if z = 0, g(z)= 1/2 and <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+g%5E%5Cprime+%28z%29+%3D+-1%2F4" 
alt="g^\prime (z) = -1/4">

##### For tanh
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%0Ag%28z%29+%3D+tanh%28z%29+%3D++a++%3D+%5Cfrac+%7Be%5E%7Bz%7D+-+e%5E%7B-z%7D%7D%7Be%5E%7Bz%7D+-+e%5E%7B-z%7D%7D+" 
alt="
g(z) = tanh(z) =  a  = \frac {e^{z} - e^{-z}}{e^{z} - e^{-z}} ">

now, taking derivative

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%0Ag%5E%5Cprime+%28z%29+%3D+1+-+%28tanh+%28z%29%29+%5E%7B2%7D+%3D+1+-+a%5E%7B2%7D+" 
alt="
g^\prime (z) = 1 - (tanh (z)) ^{2} = 1 - a^{2} ">

again, if z is large(+ve or -ve), g(z)=tanh(z) will be close to 1 and <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+g%5E%5Cprime+%28z%29+%3D+0+" 
alt="g^\prime (z) = 0 ">

if z = 0, g(z)= 1/2 and <img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%0Ag%5E%5Cprime+%28z%29+%3D+1" 
alt="
g^\prime (z) = 1">

##### For ReLU and Leaky ReLU functions

![rel](https://i.ibb.co/4YzTZwD/der.png)

### Gradient Descent for Neural Networks

Take Paramters <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++w%5E%7B%5B1%5D%7D+%2C+%5C%2C+b%5E%7B%5B1%5D%7D+%2C+%5C%2C+w%5E%7B%5B2%5D%7D+%2C+%5C%2C+b%5E%7B%5B2%5D%7D+%5C%3B" 
alt=" w^{[1]} , \, b^{[1]} , \, w^{[2]} , \, b^{[2]} \;">
we have, 
- n<sub>x</sub> = n<sup>[0]</sup> input features 
- n<sup>[1]</sup>hidden units, and 
- n<sup>[2]</sup>output 

For a binary classification
Cost Function :<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+J+%28w%5E%7B%5B1%5D%7D+%2C+%5C%2C+b%5E%7B%5B1%5D%7D+%2C+%5C%2C+w%5E%7B%5B2%5D%7D+%2C+%5C%2C+b%5E%7B%5B2%5D%7D%29+%3D+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%5Cmathcal+L%28%5Chat+y+%2C+y%29+" 
alt="J (w^{[1]} , \, b^{[1]} , \, w^{[2]} , \, b^{[2]}) = \frac{1}{m} \sum_{i=1}^{m} \mathcal L(\hat y , y) ">

Applying Gradient Descent, 

>Repeat
      Compute Prediction (ŷ, . . . \, i-1 . . . , \, m )
     <img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AdW%5E%7B%5B1%5D%7D+%3D+%5Cfrac+%7BdJ%7D%7BdW%5E%7B%5B1%5D%7D%7D++%5C%3B+%5C%3B++%3B+%5C%3B+%5C%3B+dW%5E%7B%5B2%5D%7D+%3D+%5Cfrac+%7BdJ%7D%7BdW%5E%7B%5B2%5D%7D%7D" 
alt="
dW^{[1]} = \frac {dJ}{dW^{[1]}}  \; \;  ; \; \; dW^{[2]} = \frac {dJ}{dW^{[2]}}">
      W<sup>[1]</sup> = W<sup>[1]</sup> - α dW<sup>[1]</sup>  #update W<sup>[1]</sup>
      b<sup>[1]</sup> = b<sup>[1]</sup> - α db<sup>[1]</sup>  #update b<sup>[1]</sup>

#### FORMULAE FOR CALCULATING COMPUTING DERIVATIVES

**Forward Propogation**

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AZ%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7DX+%2B+b%5E%7B%5B1%5D%7D+" 
alt="
Z^{[1]} = W^{[1]}X + b^{[1]} ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AA%5E%7B%5B1%5D%7D+%3D+g%5E%7B%5B1%5D%7D%28Z%5E%7B%5B1%5D%7D%29+" 
alt="
A^{[1]} = g^{[1]}(Z^{[1]}) ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0A+Z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7DA%5E%7B%5B1%5D%7D+%2B+b%5E%7B%5B2%5D%7D+" 
alt="
 Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AA%5E%7B%5B2%5D%7D+%3D+g%5E%7B%5B2%5D%7D%28Z%5E%7B%5B2%5D%7D%29+%3D+%5Csigma+%28Z%5E%7B%5B2%5D%7D%29+" 
alt="
A^{[2]} = g^{[2]}(Z^{[2]}) = \sigma (Z^{[2]}) ">

**Back Propogation**

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AdZ%5E%7B%5B2%5D%7D+%3D+A%5E%7B%5B2%5D%7D+-+Y+" 
alt="
dZ^{[2]} = A^{[2]} - Y ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0AdW%5E%7B%5B2%5D%7D+%3D+%5Cfrac+%7B1%7D%7Bm%7D+dZ%5E%7B%5B2%5D%7D+A%5E%7B%5B1%5DT%7D+" 
alt="
dW^{[2]} = \frac {1}{m} dZ^{[2]} A^{[1]T} ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0Adb%5E%7B2%7D+%3D+%5Cfrac+%7B1%7D%7Bm%7D+np.sum+%28dZ%5E%7B%5B2%5D%7D+%2C+axis%3D1%2C+keepdims+%3D+true%29+" 
alt="
db^{2} = \frac {1}{m} np.sum (dZ^{[2]} , axis=1, keepdims = true) ">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0A+dZ%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B2%5DT%7D+dZ%5E%7B2%7D+%2A+%28g%5E%7B%5B1%5D%7D%29%5E%5Cprime+%28Z%5E%7B%5B1%5D%7D%29+" 
alt="
 dZ^{[1]} = W^{[2]T} dZ^{2} * (g^{[1]})^\prime (Z^{[1]}) ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0A+dW%5E%7B%5B1%5D%7D+%3D+%5Cfrac+%7B1%7D%7Bm%7D+dZ%5E%7B%5B1%5D%7D+X%5E%7BT%7D+" 
alt="
 dW^{[1]} = \frac {1}{m} dZ^{[1]} X^{T} ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%0Adb%5E%7B1%7D+%3D+%5Cfrac+%7B1%7D%7Bm%7D+np.sum+%28dZ%5E%7B%5B1%5D%7D+%2C+axis%3D1%2C+keepdims+%3D+true%29+" 
alt="
db^{1} = \frac {1}{m} np.sum (dZ^{[1]} , axis=1, keepdims = true) ">

In Summary
![sum](https://i.ibb.co/Qkz65HS/gradides.png)

### Random Initialization

We can't initialize W weights as 0, but we can initialize b as 0
![0](https://i.ibb.co/x61XYjQ/image.png)

**How to initialize weights randomly**
```
W<sup>[1]</sup> = np.random.randn((2,2)) * (0.01) 
b<sup>[1]</sup> = np.zero((2,1)) 
W<sup>[2]</sup> = np.random.randn((1,2)) * (0,01) 
b<sup>[2]</sup> = 0 
```
we multiply by a small number, so that in an activation function so that we get a proper differential value (W remains small hence)
For a shallow NN, 0.01 makes do, but for deeper NN, a smaller Constant must be used

(Week 4)
---
## Deep Neural Networks
---

### Deep L-Layer Neural Network

Introduction + Recapping

![dnn](https://i.ibb.co/2h6hxZr/dnn.png)

Notation

![note](https://i.ibb.co/8sY0FvP/notation.png)

In a deep neural network

**L** is the number of Layers , **n<sup>[l]</sup>** = Number of Units in Layer l
**a<sup>[l]</sup>** are the number of activations in Layer l

### Forward Propogation in a Deep Neural Network

![deep](https://i.ibb.co/bQz8vpM/deep.png)
with single training example X

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+X+%3A+Z%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7D+X+%2B+b%5E%7B%5B1%5D%7D%0A" 
alt="X : Z^{[1]} = W^{[1]} X + b^{[1]}
">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B1%5D%7D+%3D+g%5E%7B%5B1%5D%7D%28z%5E%7B%5B1%5D%7D%29+" 
alt="a^{[1]} = g^{[1]}(z^{[1]}) ">

General Rule for this, 

![general](https://i.ibb.co/5TpmrL2/general.png)

>Vectorized we get,
where,
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+X+%3D+A%5E+%7B%5B0%5D%7D" 
alt="X = A^ {[0]}">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+X+%3A+Z%5E%7B%5B1%5D%7D+%3D+W%5E%7B%5B1%5D%7D+X+%2B+b%5E%7B%5B1%5D%7D%0A" 
alt="X : Z^{[1]} = W^{[1]} X + b^{[1]}
">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+a%5E%7B%5B1%5D%7D+%3D+g%5E%7B%5B1%5D%7D%28z%5E%7B%5B1%5D%7D%29+" 
alt="a^{[1]} = g^{[1]}(z^{[1]}) ">
and
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+Z%5E%7B%5B2%5D%7D+%3D+W%5E%7B%5B2%5D%7D+A%5E%7B%5B1%5D%7D+%2B+b%5E%7B%5B1%5D%7D+" 
alt="Z^{[2]} = W^{[2]} A^{[1]} + b^{[1]} ">
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+A%5E%7B%5B2%5D%7D+%3D+g%5E%7B%5B2%5D%7D%28Z%5E%7B%5B2%5D%7D%29+" 
alt="A^{[2]} = g^{[2]}(Z^{[2]}) ">
so, finally
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle++%5Chat+y+%3D+g%28Z%5E%7B%5B4%5D%7D%29+%3D+A%5E%7B%5B4%5D%7D+" 
alt=" \hat y = g(Z^{[4]}) = A^{[4]} ">

from general rule
(taking from l=1 to l=4)
>> **HERE FOR LOOP IS FINE , EVEN AFTER VECTORIZING**

### A few applications for deep neural networks

- Edge Detection leading to Feature Detection in Images
  
  ![face](https://i.ibb.co/ssXxdm5/face.png)
  
- Language and Speech Recognition 
Audio - > Low Level Audio Waveform Features - > Phonemes (Sound Units) - > Words - > Sentences 

Example :-  **Circuit theory and deep learning**

Informally : There are functions you can compute with a small L-layer deep neural network that shallower networks require exponentially more hidden units to complete

### Building Blocks of Deep Neural Networks

Taking forward and backward functions in a neural network

![de](https://i.ibb.co/N6CnM65/fwdbk.png)

Here, we use the standard convention for a L-Layered Neural Network
we take Input a<sup>[l-1]</sup> and output a<sup>[l]</sup>, and we also cache z<sup>[l]</sup> for forward, and da<sup>[l]</sup> as input and da<sup>[l-1]</sup> as output for backward

Using normal formulae for both Forward and Backward Propopgation, applying proper activation functions, we get whatever result we require.

If we implement these two functions, 

this is what we can expect to happen, graphically
![gr](https://i.ibb.co/98DgGy0/applying.png)

here, input a<sup>[0]</sup> is given, we pass it through various activation functions, get intermediate output values, cache the z<sup>[l]</sup> values for easy calculation for both W and b and also for easier backpropogation function calculation of derivatives.
This is one iteration of gradient descent for the neural network

Eventually we calculate the value of ŷ

### Forward and Backward Propogation

#### Forward Propogation for layer l
- Input a<sup>[l-1]</sup>
- Output a<sup>[l]</sup> 
- Cache z<sup>[l]</sup>
  
![g](https://i.ibb.co/Lg6YcC2/fwd.png)

#### Backward Propogation for layer l
- Input da<sup>[l]</sup>
- Output da<sup>[l-1]</sup> , dW<sup>[l]</sup> , db<sup>[l]</sup>

![g](https://i.ibb.co/GnJtVxC/bk.png)

### Parameters and HyperParameters

Parameters  : W<sup>[1]</sup> ,  W<sup>[2]</sup>  ,  b<sup>[l]</sup>  ,  b<sup>[2]</sup> . . . etc

HyperParameters : 
- Learning Rate α , 
- Number of Iterations , 
- Number of Hidden Layers (L) , 
- Number of Hidden Units (n<sup>[1]</sup> , n<sup>[2]</sup)
- Choice of Activation Functions

---