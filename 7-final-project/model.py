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

def get_device():
    """Return the GPU device if available otherwise CPU"""
    torch_device = torch.device(["cpu", "cuda"][torch.cuda.is_available()])
    return torch_device

## Set up the classifier and model

# Instantiate and configure pytorch classifier
# using VGG classifier feature settings
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 102)),
    ('dropout', nn.Dropout(0.2)),
    ('output', nn.LogSoftmax(dim = 1))
]))

# Select torchvision model
model = models.vgg16(pretrained = True)

# Freeze parameters so to avoid backpropagating
for param in model.parameters():
    param.requires_grad = False

# Attach classifier to pretrained model
model.classifier = classifier


## Train and validate the model

device = get_device()

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
            # output the loss     
            print(f"""
                Epoch {epoch + 1}/{epochs} -- 
                Training loss: {running_loss / print_every:.3f} -- 
                Validation loss: {valid_loss / len(valid_loader):.3f} -- 
                Validation accuracy: {accuracy / len(valid_loader):.3f}
            """)
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
        predicted = torch.max(forward_results.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().float().item()
    print(f"Test accuracy: {correct / total}")


## Save and rebuild the model

# Save a model checkpoint
def save_model(model, optimizer):
    """Save a checkpoint for the model."""
    model.class_to_idx = train_dataset.class_to_idx
    device = get_device()
    model.to(device)
    checkpoint = {
        'classifier': model.classifier,
        'features': model.features,
        'optimizer': optimizer.state_dict(),
        'input_size': 25088,
        'output_size': 102,
        'state_dict': model.state_dict(),
        'idx_to_class': model.class_to_idx
    }
   torch.save(checkpoint, 'checkpoint.pt')
   print("Model saved.")
   return checkpoint

save_model(model, optimizer)

# Load a saved checkpoint
def load_checkpoint(file_path):
    """Rebuild the model from a saved checkpoint."""
    device = get_device()
    model_data = torch.load(file_path)
    model = models.vgg16(pretrained = True)
    model.classifier = model_data.get('classifier', {
        ('fc1', nn.Linear(25088, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 102)),
        ('dropout', nn.Dropout(0.2)),
        ('output', nn.LogSoftmax(dim = 1))
    })
    model.class_to_idx = model_data.get('idx_to_class')
    model.load_state_dict(model_data['state_dict'])
    model.to(device)
    print("Model loaded.")
    return (model, model_data)

model, model_data = load_checkpoint('checkpoint.pt')


## Preprocess image to use as model input
## This is used when making predictions below.

# preprocessor function
def process_image(image, size=256):
    """Scales, crops, and normalizes a PIL image for a PyTorch model.
    Returns a Numpy array.
    """
    # Load and set up image
    pil_image = Image.open(image)

    # # NOTE: below attempted to follow instructions on using and adjusting PIL image.
    # # These attempts were all unsuccessful. Just using transforms.Compose
    # # seemed to work fine, but does not avoid the issues raised in the instructions!
    #
    # "You'll want to use PIL to load the image (documentation). It's best to write a
    # function that preprocesses the image so it can be used as input for the model.
    # This function should process the images in the same manner used for training.
    # "First, resize the images where the shortest side is 256 pixels, keeping the
    # aspect ratio. This can be done with the thumbnail or resize methods. Then you'll
    # need to crop out the center 224x224 portion of the image.
    # Color channels of images are typically encoded as integers 0-255, but the model
    # expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy
    # array, which you can get from a PIL image like so np_image = np.array(pil_image).
    # As before, the network expects the images to be normalized in a specific way.
    # For the means, it's [0.485, 0.456, 0.406] and for the standard deviations
    # [0.229, 0.224, 0.225]. You'll want to subtract the means from each color channel,
    # then divide by the standard deviation.
    # And finally, PyTorch expects the color channel to be the first dimension but
    # it's the third dimension in the PIL image and Numpy array. You can reorder
    # dimensions using ndarray.transpose. The color channel needs to be first and retain
    # the order of the other two dimensions."
    #
    # # resize image
    # width, height = pil_image.size
    # if width > height:
    #     height = size
    #     width *= (size / height)
    # else:
    #     width = size
    #     height *= (size / height)
    # pil_image.thumbnail((width, height))
    # # normalize color values
    # np_image = np.array(pil_image)
    # np_image = np_image.astype('float32')
    # np_image /= 255.0
    # print(np_image.size)
    #
    # # PyTorch expects the color channel to be the first dimension.
    # # It's the third dimension in the PIL image and Numpy array.
    # # Reorder dimensions using ndarray.transpose. The color channel needs
    # # to be first; retain the order of the other two dimensions.
    # image_a = image_a.transpose(2,0,1)

    # Transform into image tensor
    transformed_image = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])(pil_image) #(np_image)

    return transformed_image

# check the preprocessor function
def imshow(image, ax=None, title=None):
    """Imshow for Tensor. Checks that preprocessing worked by
    reversing the process and returning original (cropped) image"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes it is the third dimension
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
## This is where you pass in an image and make predictions!

# Predict the top 5 or so (top-𝐾) most probable classes
# in the tensor (x.topk(k))
def predict(image_tensor, model, k=5):
    """Returns the top 𝐾 most likely classes for an image along with the
    probabilities. This method returns both the highest probabilities and
    the indices of those probabilities corresponding to the classes.
    Args:
        image_path: path to an image file
        model:      saved model checkpoint
        k:          number of top predictions to return
    Returns:
        k highest probabilities and their class labels
    """
    # Calculate the class probabilities then find the 𝐾 largest values.
    
    # Turn off gradients to speed up this part
    with torch.no_grad():
        log_ps = model.forward(image_tensor)
        ps = torch.exp(log_ps)
        top_ks = ps.topk(k, dim = 1)
    top_ps = top_ks[0][0]
    top_classes = top_ks[1][0]
    
    # Build a list of labels and probabilities
    predictions = []
    # Convert indices to class labels using model.class_to_idx (added earlier);
    # Invert the dict to get a mapping from index to class.
    idx_to_label = {
        v: k for k, v in model.class_to_idx.items()
    }
    for i in range(len(top_ps)):
        # Get index and convert index to class label
        top_idx = int(top_classes[i])
        label = idx_to_label[top_idx]
        # Get probability and store it alongside class
        probability = float(top_ps[i])
        predictions.append((label, probability)) 

    # Return chartable class labels and predictions
    return np.array(predictions).transpose()


## Display results and check visually

# Check to make sure the trained model's predictions
# make sense. Even if the testing accuracy is high,
# check that there aren't obvious bugs. Use `matplotlib`
# to plot probabilities for the top-k classes in a
# bar graph. Show the input image alongside the graph.
#
# The `cat_to_name.json` maps indexes to labels.

# Manually choose one image to load
image_path = "flowers/test/10/image_07090.jpg"
# convert integer to flower name with cat_to_name.json
image_label = "(Unidentified image)"
for label in model.class_to_idx:
    if model.class_to_idx[label] == '07090':
        image_label = label
        break
# Load an image tensor and calculate top-k predictions
image = process_image(image_path)
unsqueezed_image = image.unsqueeze(0)
#unsqueezed_image.requires_grad = False
#unsqueezed_image.to(get_device())
model.to('cpu')
model.eval()
# Get the class labels and prediction probabilities
k = 5
predictions = predict(unsqueezed_image, model, k=k)

# Display the image and graph top-k labels, probabilities
# Use previously defined `imshow` to display tensor as image
imshow(image, title="image_label")
plt.show()
# Set up top-k bar chart
labels = predictions[0]
probabilities = predictions[1]
plt.barh(labels, probabilities, align = "center")
plt.title(f"Top {k} Predictions")
plt.show()


# TODO: afterwards use argparse to create CLI 
# - run training py passing in data dir (required) and optional args:
#   - save dir, architecture, learning rate, hidden units, epochs, gpu
#   - action 'store' versus 'store_true' (for gpu) if gpu came in at all
# - run predicting py passing in img path, checkpoint path
#   - specify specific architecture
#   - hidden units, top k, category names (path to cat_to_names json)
