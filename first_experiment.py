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


experiment_dataframe = pd.DataFrame(columns=['Model', 'Optimizer', 'Test Accuracy', 'Train Accuracy'])

# This experiment compares the performance of a neural 
# network with and without dropout using different optimizers

INPUT_SIZE = 784
HIDDEN_SIZE = 1024
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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['SGD'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Stochastic Gradient Descent')
plt.savefig('first_experiment_output/SGD.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------


# With Adam
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
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['Adam'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Adam')
plt.savefig('first_experiment_output/Adam.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------
# With Adagrad
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
    optimizer = torch.optim.Adagrad(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['Adagrad'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Adagrad')
plt.savefig('first_experiment_output/Adagrad.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------
# With Adadelta
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
    optimizer = torch.optim.Adadelta(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['Adadelta'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Adadelta')
plt.savefig('first_experiment_output/Adadelta.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------
# With AdamW
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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['AdamW'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('AdamW')
plt.savefig('first_experiment_output/AdamW.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------
# With Adamax
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
    optimizer = torch.optim.Adamax(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['Adamax'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Adamax')
plt.savefig('first_experiment_output/Adamax.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------
# With RMSprop
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
    optimizer = torch.optim.RMSprop(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Training the model

    y = train_model(model, train_loader, test_loader, criterion, optimizer, n_total_steps, NUM_EPOCHS)
    result_list.append(y)

    # Testing model performance after training

    test_accuracy, test_message = test_model(model, test_loader)
    train_accuracy, train_message = test_model(model, train_loader)
    print(test_message)
    # Add row to dataframe
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Optimizer' : ['RMSprop'],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('RMSprop')
plt.savefig('first_experiment_output/RMSprop.png')
plt.clf()


experiment_dataframe.to_excel('first_experiment_output/experiment_dataframe.xlsx')
