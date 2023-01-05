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

# Importing the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='/data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Creating the data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=settings.BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=settings.BATCH_SIZE,
                                          shuffle=False)

# examples = iter(test_loader)
# example_data, example_targets = next(examples)
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(example_data[i][0], cmap='gray')
# plt.show()

# Initializing the models to be used

simple_naive_neural_net = SimpleNaiveNeuralNet(settings.INPUT_SIZE,
                                               settings.HIDDEN_SIZE,
                                               settings.NUM_CLASSES)

naive_neural_net = NaiveNeuralNet(settings.INPUT_SIZE,
                                  settings.HIDDEN_SIZE,
                                  settings.NUM_CLASSES)
dropout_neural_net = DropoutNeuralNet(settings.INPUT_SIZE,
                                      settings.HIDDEN_SIZE,
                                      settings.NUM_CLASSES)

models = [naive_neural_net, dropout_neural_net, simple_naive_neural_net]

result_list = []

for model in models:

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    n_total_steps = len(train_loader)

    # Testing model performance before training

    test_model(model, test_loader)

    # Training the model

    y = train_model(model, train_loader, criterion, optimizer, n_total_steps)
    result_list.append(y)

    # Testing model performance after training

    test_model(model, test_loader)


# Plotting the loss of all model's over the training iterations

# Build DataFrame dictionary
model_dict = {type(model).__name__: result_list[index] for index, model in enumerate(models)}
data = pd.DataFrame(model_dict)
# Plot training history
sns.set_theme()
sns.lineplot(
    data=data, linewidth=0.2
    )
plt.ylim(0, 1)
plt.show()
