# Quick Overview

NOTE: follow along in `example.py` code from Transfer Learning solution

- the math you need:
    - matrix for moving from one layer of perceptron to next
    - activation function is how to calculate final value of node
    - derivatives calculate slope to determine step direction

- perceptron: inputs, weights and bias to sum, sum to sigmoid, output
    - check out image
    - bias not always drawn as node, sometimes next to sum
    - often sum spot out because you know you're summing matrix math
    - Sigmoid scrunches 0-1 but not only activation function
- use perceptrons in layers, turn into matrix math
    - input layer is vector
    - matrix is all weights
    - this gives summation to run sigmoid function

- epoch: one cylce of feed-forward then backpropagation
    - feed-forward running forward
    - turn around running same in opposite refining weights
    - backpropagate to change weights, then run through again

- when to stop fitting to your training data ("early stopping")
    - underfitting when training data and testing data both similarly big error
    - stop fitting when training and testing error are small
    - overfitting when training error is small but testing goes back up

- node "dropout"
    - too heavily weighted node so others aren't adjusting
    - give x percent chance for that node not to work on a turn

- "gradient descent" and getting stuck in local minima
    - address with random restarts
    - run your model at different starting points

- non-sigmoid to deal with no place to step near minima
    - hyperbolic tangent as a big step forward there, -1 to 0
    - rectified linear unit as another way to get 0,1
- always end with a sigmoid though
    - on last layer you do need something between 0 and 1

- only train on some input points instead of all every time
    - rotate through subsets
    - avoid heavy computation
    - less accurate step each epoch

- "learning rate decay"
    - if slope is steep, take a long step, if shallow take short
    - big gain from big steps early on
    - as learning curve gets smaller, smaller steps down

- momentum: weight previous steps higher closer to you
    - this way you can get over bumps near local minima
    - vague but seems to work well in practice

- output nodes: number of values compared is number of outputs
    - just true or false would be one output
    - variant letters of the alphabet would be number of letters
