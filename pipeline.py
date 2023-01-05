import torch
import settings


def train_model(model, train_loader, criterion, optimizer, n_total_steps):
    y = []
    for epoch in range(settings.NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y.append(loss.item())
            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch+1}/{settings.NUM_EPOCHS}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    return y


def test_model(model, test_loader):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')