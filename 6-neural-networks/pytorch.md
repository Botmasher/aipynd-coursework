# Deep Learning with PyTorch

(Lesson 4 in neural networks)

## Intro and first problem
- tensors, autograd, validation, transfer learning
- calculate the output of a network 
```Python
import torch

def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

### generate data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))
output = activation(torch.sum(features * weights) + bias)

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
```

- create array with `np.random.rand()`, pass to `torch.from_numpy()`
    - you can still read its `.numpy()`!
    - changing in one will change values in other

## Overview of the code
- I'll comment on problems solved here and see if they come in handy in my project
- solutions in class notebooks `.../notebooks/deep-learning-v2-pytorch/intro-to-pytorch/`
1. just generate some data and calculate the output (above) 
2. deep learning network
    - identify different handwritten digits 0 to 9
    - "Create a network with 784 input units, a hidden layer with 128 units and a ReLU activation, then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above."
    - `Part 2 - Neural Networks in PyTorch (Solution).ipynb`
3. build on above, loss and walking down gradient
    - make the network brilliant instead of not so smart
    - `Part 3 - Training Neural Networks (Solution).ipynb`
4. classify images in the fashion MNIST
    - tell 28x28 pixel sneakers apart from shirts!
    - examples of building and training the network
    - `Part 4 - Fashion-MNIST (Solution).ipynb`
5. use fashion MNIST network to make predictions
    - create and train `model` instantiating our `Classifier(nn.module)`
    - infer: `model.eval()` and turn off autograd with the `torch.no_grad()`
    - `Part 5 - Inference and Validation (Solution).ipynb`
6. save and load models
    - `Part 6 - Saving and Loading Models.ipynb`
7. load images
    - `Part 7 - Loading Image Data (Solution).ipynb`
8. transfer learning
    - use a pre-trained network on data not in the dataset
    - models like ImageNet do very well on images not trained on
    - use CUDA and finish training the model
    - train pretrained models to work on cat and dog images (DenseNet or ResNet)
    - `Part 8 - Transfer Learning (Solution).ipynb`

## Notes on working through the code
- set up / load testing and training data
    - training, do things like rotation and crop
    - both must `ToTensor` and `Normalize` (rgb normalization)
    - for test usually resize to 255 and center crop to 224
    - use this later
- standard imports: pyplot, torch, torchvision (for photos)
    - torch's `nn`, `optim` and then `nn.functional as F`
    - torchvision's `datasets, transforms, models`
- CUDA is for running on GPU; this for `model.to(device)`
- pretrained classifier: `models.densenet121(pretrained=True)`
    - freeze params while working on that model though
    - this way you're not changing the pretrained model
- define the model
    - check that input matches pretrained output
    - add layers (you're adding hidden layer to trained)
    - choose activation function
    - dropout chance, then linear, then logsoftmax
- calculate your loss (`nn.NLLLoss`)
- now send to device so it's using GPU (about to train!)
- then train and test the model
    - earlier you loaded training data
    - loop through epochs, sending inputs and labels to device
    - first training step run forward and backward
    - then testing step to check how training has gone 
        - only run forward
        - do not train on testing data so no backpropagation
        - calculate the test accuracy, loss
