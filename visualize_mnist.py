import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import settings
from nets.NaiveNeuralNet import NaiveNeuralNet
from nets.DropoutNeuralNet import DropoutNeuralNet
from nets.SimpleNaiveNeuralNet import SimpleNaiveNeuralNet
from pipeline import test_model, train_model
import numpy as np

INPUT_SIZE = 784
HIDDEN_SIZE = 1024
NUM_CLASSES = 10
NUM_EPOCHS = 50
BATCH_SIZE = 100

LEARNING_RATE = 0.001

# Importing the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='/data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Creating the data loaders
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=settings.BATCH_SIZE,
#                                            shuffle=True)

# Generate random set of 5000 indices
sample_indices = np.random.choice(len(train_dataset), 5000)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           sampler=sample_indices)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False)

# Visualizing the data
examples = iter(test_loader)
example_data, example_targets = next(examples)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()