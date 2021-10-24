### Gradient Descent
>Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. The goal of the gradient descent is to minimise a given function which, in our case, is the loss function of the neural network.
>
### Batch Gradient Descent
>In Batch Gradient Descent, all the training data is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters. So that’s just one step of gradient descent in one epoch.
>
### Mini Batch Gradient Descent
>In this method, neither we use all the dataset all at once nor we use the single example at a time. We use a batch of a fixed number of training examples which is less than the actual dataset and call it a mini-batch. 
The steps involved in this are:
    1. Pick a mini-batch
    2. Feed it to Neural Network
    3. Calculate the mean gradient of the mini-batch
    4. Use the mean gradient we calculated in step 3 to update the weights
    5. Repeat steps 1–4 for the mini-batches we created
On a large data set, mini batch runs much faster than batch gradient descent.
>
![](/resources/ImprovingDeepNeuralNetworks/images/BatchvsMiniBatch.png)

>### Exponentially weighted Average
>
![](/resources/ImprovingDeepNeuralNetworks/images/EWG.png)

>Let’s say we want to calculate moving average of the temperature in the last N days. What we do is we start with 0, and then every day, we’re going to combine it with a weight of a parameter(let’s say β) times whatever is the value accumulated until now plus 1-β times that day’s temperature.  
_Vt = β * (Vt-1) + (1-β)*NewSample_
The value of 1-β is what we multiply with the current value. If we expand the equation, we see that we end up multiplying the current value by 1-β and the previous values of β are *exponentially decaying* on the curve.
>
### Bias correction

>Making EWMA more accurate — Since the curve starts from 0, there are not many values to average on in the initial days. Thus, the curve is lower than the correct value initially and then moves in line with expected values.
>
![](/resources/ImprovingDeepNeuralNetworks/images/Bias.png)
>
>In the figure above, when we use the formula before, we get the purple curve whereas we should be getting the green curve in actual scenario. To get the required curve, we implement the Bias correction as follows:
>
>Example: Starting from t=0 and moving forward,
        Vsub0 = 0Vsub1 = 0.98Vsub 0+0.02θsub1 = 0.020θsub1
        Vsub2 = 0.98Vsub1 + 0.02θsub2 = 0.0196θsub1+0.02θsub2
The initial values of Vt will be very low which need to be compensated.
 _*Make Vt = Vt/1−βsupt*_
>    for t=2, 1−βsupt= 1−0.9⁸² = 0.0396 (Bias Correction Factor)
    Vsub2 = V2/0.0396 = 0.0196θsub1 + 0.02θsub2 / 0.0396
When t is large, 1/1−βsupt =1, hence bias correction factor has no effect when t is sufficiently large.

>### Gradient Descent with Momentum

>Gradient descent with momentum will always work much faster than the algorithm Standard Gradient Descent. The basic idea of Gradient Descent with momentum is to calculate the exponentially weighted average of your gradients and then use that gradient instead to update your weights.It functions faster than the regular algorithm for the gradient descent.
>
#### Implementation:
>We use dW and db to update our parameters W and b during the backward propagation as follows:
    W = W — learning rate * dW
    b = b — learning rate * db
In momentum we take the exponentially weighted averages of dW and db, instead of using dW and db independently for each epoch.
    VdW = β * VdW + (1 — β) * dW
    Vdb = β * Vdb + (1 — β) *db

>#### Adam optimization:
>
>Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent. The method is really efficient when working with large problem involving a lot of data or parameters.
More about the optimization can be found out by clicking [here](https://www.geeksforgeeks.org/intuition-of-adam-optimizer/)

>### Learning Rate Decay
>Learning rate decay is a technique for training modern neural networks. It starts training the network with a large learning rate and then slowly reducing/decaying it until local minima is obtained. It is empirically observed to help both optimization and generalization.

>When we have a constant learning rate, the steps taken by our algorithm while iterating towards minima are so noisy that after certain iterations it seems wandering around the minima and do not actually converges.
>
>When the learning rate is large initially we still have relatively fast learning but as tending towards minima learning rate gets smaller and smaller, end up oscillating in a tighter region around minima rather than wandering far away from it.
>
>Learing rate decay is implemented as follows:
*_α=(1/(1+decayRate×epochNumber))*​α0 _*



