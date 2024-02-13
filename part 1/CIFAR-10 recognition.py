import torch
import torch.optim as optim  # Backpropagation and forward step optimizers
import torch.nn as nn
from collections import OrderedDict  # Layer naming in sequential models
from matplotlib import pyplot as plt  # Plotting
from torchvision import datasets  # Importing and accesses CIFAR-10
from torchvision import transforms  # Transforming luna to PyTorch tensor

# # DATA
data_path = '../../data-unversioned/p1ch7/'

cifar10_means = (0.4914, 0.4822, 0.4465)
cifar10_stds = (0.2470, 0.2435, 0.2616)
cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True,
    transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(cifar10_means, cifar10_stds)  # Normalized RGB
    ])
)
cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True,
    transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize(cifar10_means, cifar10_stds)  # Normalized RGB
    ])
)

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in cifar10 if label in [0, 2]]
cifar2_val = [(img, label_map[label]) for img, label in cifar10_val if label in [0, 2]]


# # NEURAL NETWORK MODEL
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

n_out = 2  # one-hot encoding classification
model = nn.Sequential(OrderedDict([
    ('hidden_layer', nn.Linear(3072, 512)),  # 3 x 32 x 32 = 3072 input --> 512 hidden layer
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(512, n_out)),
    ('softmax', nn.LogSoftmax(dim=1))  # Log fixes the tend to infin as --> 0
]))

loss_fn = nn.NLLLoss()  # Loss function

# # HYPERPARAMETERS
n_epochs = 100
learning_rate = 1e-2
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# # TRAINING LOOP
for epoch in range(1, n_epochs + 1):
    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))  # LogSoftmax output from model
        loss = loss_fn(outputs, labels)  # takes in the (2, 1)-tensor output and (1)-tensor label = 0 or 1

        optimizer.zero_grad()
        loss.backward()  # Backpropagate to find dloss / dw1, dloss / dw2, etc.
        optimizer.step()

    print(f"Epoch {epoch}, Training Loss {float(loss)},")

# # final accuracy test
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=64, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        batch_size = imgs.shape[0]
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())

print(f"Accuracy: {correct / total}")

numel_list = [p.numel() for p in model.parameters() if p.requires_grad is True]
print(numel_list)

# # PLOTTING

# t_range = torch.arange(20., 90.).unsqueeze(1)
#
# fig = plt.figure(dpi=600)
# plt.xlabel("Temperature (°Fahrenheit)")
# plt.ylabel("Temperature (°Celsius)")
#
# plt.plot(*zip(*sorted(zip(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy()))), 'c-')  # Predicted
# plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
# plt.plot(t_u.numpy(), t_c.numpy(), 'o')  # Actual
# plt.show()
