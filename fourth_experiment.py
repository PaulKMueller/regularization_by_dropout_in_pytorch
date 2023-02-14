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

experiment_dataframe = pd.DataFrame(columns=['Model', 'Number of Epochs', 'Test Accuracy', 'Train Accuracy'])

INPUT_SIZE = 784
HIDDEN_SIZE = 1024
NUM_CLASSES = 10
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



NUM_EPOCHS = 50

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 50')
plt.savefig('fourth_experiment_output/50.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------


NUM_EPOCHS = 100

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 100')
plt.savefig('fourth_experiment_output/100.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

NUM_EPOCHS = 1000

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 1000')
plt.savefig('fourth_experiment_output/1000.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

NUM_EPOCHS = 2000

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 2,000')
plt.savefig('fourth_experiment_output/2000.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

NUM_EPOCHS = 3000

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 3,000')
plt.savefig('fourth_experiment_output/3000.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

NUM_EPOCHS = 4000

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 4,000')
plt.savefig('fourth_experiment_output/4000.png')
plt.clf()



# -----------------------------------------------------------------------------------------------------------

NUM_EPOCHS = 5000

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
    experiment_dataframe = pd.concat([experiment_dataframe, pd.DataFrame({'Model': [type(model).__name__], 'Number of Epochs' : [NUM_EPOCHS],'Test Accuracy' : [test_accuracy], 'Train Accuracy': [train_accuracy]})])


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
plt.title('Number of epochs = 5,000')
plt.savefig('fourth_experiment_output/5000.png')
plt.clf()

experiment_dataframe.to_excel('fourth_experiment_output/experiment_dataframe.xlsx')

# -----------------------------------------------------------------------------------------------------------