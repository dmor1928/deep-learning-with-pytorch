import torch
import torch.optim as optim  # Backpropagation and forward step optimizers
import torch.nn as nn
from collections import OrderedDict  # Layer naming in sequential models
from matplotlib import pyplot as plt  # Plotting
from torchvision import datasets  # Importing and accesses CIFAR-10
from torchvision import transforms  # Transforming luna to PyTorch tensor
import torch.nn.functional as F  # For functionals in Net class
import datetime  # Print time in training

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

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print(f"Training on device {device}.")


# # MODEL
class Net(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(n_chans1 * 8 * 8 // 2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)  # 32 x 32 --> 16 x 16
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)  # 16 x 16 --> 8 x 8
        out = out.view(-1, self.n_chans1 * 8 * 8 // 2)  # N x 8 x 8 x 8 --> N x 512 (feature layers squashed to 1d vec)
        out = torch.tanh(self.fc1(out))  # Takes output from 2nd pooling (w/ 2nd pooled output reshaped)
        out = self.fc2(out)
        return out


class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)  # 2d batch-norm for n_chans1 feature layers
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)  # 2d batch-norm for n_chans//2 feature layers
        self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans1 // 4, kernel_size=3, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 4)
        self.fc1 = nn.Linear((n_chans1 // 4) * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))  # batch-normalize 1st convolution output
        out = F.max_pool2d(nn.ReLU()(out), 2)  # 32 x 32 --> 16 x 16
        out = self.conv2_batchnorm(self.conv2(out))  # batch-normalize 2nd convolution output
        out = F.max_pool2d(nn.ReLU()(out), 2)  # 16 x 16 --> 8 x 8
        out = self.conv3_batchnorm(self.conv3(out))  # batch-normalize 2nd convolution output
        out = F.max_pool2d(nn.ReLU()(out), 2)  # 8 x 8 --> 4 x 4
        out = out.view(-1, self.n_chans1 * 4)  # N x 4 x 4 x 4 --> N x 512 (feature layers squashed to 1d vec)
        out = nn.ReLU()(self.fc1(out))  # Takes output from 2nd pooling (w/ 2nd pooled output reshaped)
        out = self.fc2(out)
        return out


def parameter_count(model):
    numel_list = [p.numel() for p in model.parameters() if p.requires_grad is True]
    return sum(p.numel() for p in model.parameters()), numel_list


def save_model(model, data_path, name):
    torch.save(model.state_dict(), data_path + name)


def load_model(model, data_path, name, device):
    loaded_model = model.to(device=device)
    loaded_model.load_state_dict(torch.load(data_path + name, map_location=device))
    return loaded_model


# # TRAINING LOOP
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs)  # Model output
            loss = loss_fn(outputs, labels)  # takes in the (2, 1)-tensor output and (1)-tensor label = 0 or 1

            # l2_lambda = 0.001  # L2 Regularization
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = loss + l2_lambda * l2_norm

            optimizer.zero_grad()  # Removes accumulated grads from last batch
            loss.backward()  # Backpropagate to find grad dloss / dw1, dloss / dw2, etc.

            optimizer.step()  # Updates model parameters

            loss_train += loss.item()  # Sums losses over epoch (.item() tensor --> normal number

        if epoch == 1 or epoch % 10 == 0:
            print("{} Epoch {}, (avg per batch) Training Loss {},".format(
                datetime.datetime.now(), epoch, float(loss_train / len(train_loader))
            ))
            with torch.no_grad():
                validate(model.eval(), train_loader, val_loader)


def validate(model, train_loader, val_loader):
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)

                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)  # Index of highest value output (0 or 1) (prediction)
                total += labels.shape[0]  # Keeps count of number of examples (length of val_loader) by batch
                correct += int((predicted == labels).sum())  # Keeps count of correct predictions by batch
        print("Accuracy {}: {:.2f}".format(name, correct / total))


# # ACTUAL RUNNING
train_loader = torch.utils.data.DataLoader(cifar2, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar2_val, batch_size=128, shuffle=False)

model = NetBatchNorm(n_chans1=12).to(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=5 * 0.01)
loss_fn = nn.CrossEntropyLoss()

training_loop(
    n_epochs=400,
    optimizer=optimizer,
    model=model.train(),
    loss_fn=loss_fn,
    train_loader=train_loader,
)
validate(model.eval(), train_loader, val_loader)
save_model(model, data_path, 'birds_vs_airplanes.pt')


# # PLOTTING
