import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# for classifier setup
from collections import OrderedDict
# for converting between image indexes and labels
import json
# for preprocessing
from PIL import Image

## Data ##

# training, testing and validation data included alongside notebook
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets
# NOTE: the pre-trained networks require 224 x 224 pixels input

# Randomize scale, rotation, flip to help network generalize
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Scale, crop, normalize new testing and validation data
test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)

# Use the image datasets and transforms to define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)

# Label mapping: `cat_to_name.json` mapping encoded categories to flower names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


## Training ##

## Set up the classifier and model

# Instantiate and configure pytorch classifier
classifier = nn.Sequential(OrderedDict([
  ('fc1', nn.Linear(1024, 500)),
  ('relu', nn.ReLU()),
  ('fc2', nn.Linear(500, 2)),
  ('output', nn.LogSoftmax(dim = 1))
]))

# Select torchvision model
model = models.vgg16(pretrained = True)

# Freeze parameters so to avoid backpropagating
for param in model.parameters():
    param.requires_grad = False

# Attach classifier to pretrained model
model.classifier = classifier

# VGG classifier feature settings
# Sequential(
#    (fc1): Linear(in_features=1024, out_features=500, bias=True)
#    (relu): ReLU()
#    (fc2): Linear(in_features=500, out_features=2, bias=True)
#    (output): LogSoftmax()
# )
## print(model)

## Train and validate the model

# Default to GPU when available
device = torch.device(("cpu", "cuda:0")[torch.cuda.is_available()])

# Setup negative log probability for log softmax
criterion = nn.NLLLoss()

# Train the classifier parameters; feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)

model.to(device)

# Run through training epochs
epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # output loss     
            print(
                f"Epoch {epoch + 1}/{epochs}"
                f"Training loss: {running_loss / print_every:.3f}"
                f"Validation loss: {valid_loss / len(valid_loader):.3f}"
                f"Validation accuracy: {accuracy / len(valid_loader):.3f}"
            )
            running_loss = 0
            model.train()

## Test the model to verify post-training model accuracy

# Run through test dataset following pytorch tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        forward_results = model.forward(inputs)
        predicted = torch.max(forward_results.data, 1)[0]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test accuracy: {100 * correct / total}%")


## Save and rebuild the model

def save_model(model, optimizer):
    """Save a checkpoint for the model."""
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint = {
        'architecture': 'densenet',
        'input_size': 25088,
        'output_size': 102,
        'features': model.features,
        'classifier': model.classifier,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'idx_to_class': model.class_to_idx
    }
   torch.save(checkpoint, 'checkpoint.pth')
   print("Model saved")

save_model(model, optimizer)

# Load a saved checkpoint
def load_checkpoint(filepath):
    """Rebuild the model from a saved checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_data = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    default_classifier = nn.Sequential(
       nn.Linear(1024, 512),
       nn.ReLU(),
       nn.Linear(512, 256),
       nn.ReLU(),
       nn.Dropout(0.2),
       nn.Linear(256, 102),
       nn.LogSoftmax(dim = 1)
    )
    classifier = model_data.get('classifier', default_classifier)
    model.classifier = classifier
    model.class_to_idx = model_data.get('idx_to_class')
    model.load_state_dict(model_data['state_dict'])
    model.to(device)
    return (model, model_data)

model, model_data = load_checkpoint('checkpoint.pth')


## Preprocess image to use as model input

# preprocessor function
def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model.
    Returns a Numpy array.
    """
    open_image = Image.open(image)
    preprocess_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(40),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    return preprocess_transforms(open_image)

# check the preprocessor function
def imshow(image, ax=None, title=None):
    """Imshow for Tensor. Checks that preprocessing worked by
    reversing the process and returning original (cropped) image"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


## Predict with the model

# TODO: predict the top 5 or so (top-𝐾) most probable classes
# in the tensor (x.topk(k))
def predict_top_k(image_path, model, k=5):
    """Calculate the class probabilities then find the 𝐾 largest values.
    This method returns both the highest k probabilities and the indices
    of those probabilities corresponding to the classes.
    Args:
        image_path: path to an image file
        model:      saved model checkpoint
        k:          number of top predictions to return
    Returns:
        k highest probabilities and their class labels
    """
    # NOTE:
    # Convert indices to class labels using model.class_to_idx (added earlier)
    # Invert the dict to get a mapping from index to class.
probabilities, classes = predict_top_k(image_path, model)


## Display results and check visually

# TODO:
# Check to make sure the trained model's predictions
# make sense. Even if the testing accuracy is high,
# check that there aren't obvious bugs. Use `matplotlib`
# to plot probabilities for the top-k classes in a
# bar graph. Show the input image alongside the graph.
#
# Recall that `cat_to_name.json` maps indexes to labels.

# TODO:
# Use previously defined `imshow` to display a tensor
# as an image.
