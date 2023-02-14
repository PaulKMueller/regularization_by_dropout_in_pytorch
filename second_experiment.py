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
from pipeline import test_model, train_model
import numpy as np


# -----------------------------------------------------------------------------------------------------------

experiment_dataframe = pd.DataFrame(columns=['Model', 'Number of Neurons', 'Test Accuracy', 'Train Accuracy'])

# With 256 hidden units per hidden layer

INPUT_SIZE = 784
HIDDEN_SIZE = 256
NUM_CLASSES = 10
NUM_EPOCHS = 3000
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

# Generate random set of 1000 indices
sample_indices = np.random.choice(len(train_dataset), 1000)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           sampler=sample_indices)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)


# With 256 neurons per hidden layer
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
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Neurons' : [HIDDEN_SIZE],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Number of hidden neurons = 256')
plt.savefig('second_experiment_output/256.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------


# With 512 neurons per hidden layer
HIDDEN_SIZE = 512
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
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Neurons' : [HIDDEN_SIZE],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Number of hidden neurons = 512')
plt.savefig('second_experiment_output/512.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

# With 1024 neurons per hidden layer
HIDDEN_SIZE = 1024
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
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Neurons' : [HIDDEN_SIZE],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Number of hidden neurons = 1024')
plt.savefig('second_experiment_output/1024.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

# With 2048 neurons per hidden layer
HIDDEN_SIZE = 2048
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
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Neurons' : [HIDDEN_SIZE],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Number of hidden neurons = 2048')
plt.savefig('second_experiment_output/2048.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

# With 4096 neurons per hidden layer
HIDDEN_SIZE = 4096
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
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Neurons' : [HIDDEN_SIZE],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Number of hidden neurons = 4096')
plt.savefig('second_experiment_output/4096.png')
plt.clf()

experiment_dataframe.to_excel('second_experiment_output/experiment_dataframe.xlsx')
