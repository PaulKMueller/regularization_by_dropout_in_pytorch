import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nets.NaiveNeuralNet import NaiveNeuralNet
from nets.DropoutNeuralNet import DropoutNeuralNet
from pipeline import test_model, train_model
import numpy as np


# ----------------------------------------------------------------------------------------


INPUT_SIZE = 784
HIDDEN_SIZE = 1024
NUM_CLASSES = 10
NUM_EPOCHS = 3000
BATCH_SIZE = 100
LEARNING_RATE = 0.000001

# Importing the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='/data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Generate random set of 1000 indices
sample_indices = np.random.choice(len(train_dataset), 1000)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           sampler=sample_indices)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


# With stochastic gradient descent
# Initializing the models to be used
naive_neural_net = NaiveNeuralNet(INPUT_SIZE,
                                  HIDDEN_SIZE,
                                  NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(INPUT_SIZE,
                                      HIDDEN_SIZE,
                                      NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training
    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    print(test_model(model, test_loader)[1])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
plt.subplot(3, 2, 1)
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')


# ----------------------------------------------------------------------------------------

LEARNING_RATE = 0.00001

# With stochastic gradient descent
# Initializing the models to be used
naive_neural_net = NaiveNeuralNet(INPUT_SIZE,
                                  HIDDEN_SIZE,
                                  NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(INPUT_SIZE,
                                      HIDDEN_SIZE,
                                      NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training
    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    print(test_model(model, test_loader)[1])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
plt.subplot(3, 2, 2)
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')


# ----------------------------------------------------------------------------------------

LEARNING_RATE = 0.0001

# With stochastic gradient descent
# Initializing the models to be used
naive_neural_net = NaiveNeuralNet(INPUT_SIZE,
                                  HIDDEN_SIZE,
                                  NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(INPUT_SIZE,
                                      HIDDEN_SIZE,
                                      NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training
    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    print(test_model(model, test_loader)[1])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
plt.subplot(3, 2, 3)
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')

# ----------------------------------------------------------------------------------------

LEARNING_RATE = 0.001

# With stochastic gradient descent
# Initializing the models to be used
naive_neural_net = NaiveNeuralNet(INPUT_SIZE,
                                  HIDDEN_SIZE,
                                  NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(INPUT_SIZE,
                                      HIDDEN_SIZE,
                                      NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training
    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    print(test_model(model, test_loader)[1])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
plt.subplot(3, 2, 4)
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')


# ----------------------------------------------------------------------------------------

LEARNING_RATE = 0.01

# With stochastic gradient descent
# Initializing the models to be used
naive_neural_net = NaiveNeuralNet(INPUT_SIZE,
                                  HIDDEN_SIZE,
                                  NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(INPUT_SIZE,
                                      HIDDEN_SIZE,
                                      NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training
    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    print(test_model(model, test_loader)[1])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
plt.subplot(3, 2, 5)
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')


# ----------------------------------------------------------------------------------------

LEARNING_RATE = 0.1

# With stochastic gradient descent
# Initializing the models to be used
naive_neural_net = NaiveNeuralNet(INPUT_SIZE,
                                  HIDDEN_SIZE,
                                  NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(INPUT_SIZE,
                                      HIDDEN_SIZE,
                                      NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training
    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    print(test_model(model, test_loader)[1])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
plt.subplot(3, 2, 6)
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
