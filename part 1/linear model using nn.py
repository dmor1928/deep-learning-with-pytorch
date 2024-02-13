import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from matplotlib import pyplot as plt


def second_order_model(t_u, w1, w2, b):
    return w1 * t_u ** 2 + w2 * t_u + b


linear_model = nn.Linear(1, 1)

seq_model = nn.Sequential(OrderedDict([
    ('hidden_layer', nn.Linear(1, 12)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(12, 1))
]))


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# def calc_forward(t_u, t_c, model, loss_fn, is_train):
#     with torch.set_grad_enabled(is_train):
#         t_p = model(t_u)  # Predicted value from model
#         loss = loss_fn(t_p, t_c)  # Compared to actual value
#     return loss


def training_loop(n_epochs, optimizer, model, loss_fn, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u)
        train_loss = loss_fn(train_t_p, train_t_c)

        val_t_p = model(val_t_u)
        val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()
        train_loss.backward()  # Backpropagate to find dloss / dw1, dloss / dw2, etc.
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training Loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")


# # DATA
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1)
t_u = torch.tensor(t_u).unsqueeze(1)
t_un = 0.1 * t_u

# # Training + Validation luna splitting
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]  # Training luna
train_t_c = t_c[train_indices]

val_t_u = (t_u[val_indices])  # Validation luna
val_t_c = (t_c[val_indices])

train_t_un = (0.1 * train_t_u)  # Values are closer to zero, faster to train
val_t_un = (0.1 * val_t_u)


# # INITIALISED PARAMETERS
w = torch.zeros(())
b = torch.zeros(())

# # HYPERPARAMETERS
n_epochs = 10_000
learning_rate = 1e-3
optimizer = optim.SGD(
    seq_model.parameters(),
    lr=learning_rate)

training_loop(
    n_epochs=n_epochs,
    optimizer=optimizer,
    model=seq_model,
    loss_fn=nn.MSELoss(),
    train_t_u=train_t_un,
    val_t_u=val_t_un,
    train_t_c=train_t_c,
    val_t_c=val_t_c)


# # PLOTTING

t_range = torch.arange(20., 90.).unsqueeze(1)

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")

plt.plot(*zip(*sorted(zip(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy()))), 'c-')  # Predicted
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')
plt.plot(t_u.numpy(), t_c.numpy(), 'o')  # Actual
plt.show()
