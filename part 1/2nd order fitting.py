import torch
import torch.optim as optim
from matplotlib import pyplot as plt


def model(t_u, w1, w2, b):
    return w1 * t_u ** 2 + w2 * t_u + b


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)  # Predicted value from model
        loss = loss_fn(t_p, t_c)  # Compared to actual value
    return loss


def training_loop(n_epochs, learning_rate, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_loss = calc_forward(train_t_u, train_t_c, is_train=True)
        val_loss = calc_forward(val_t_u, val_t_c, is_train=False)

        optimizer.zero_grad()
        train_loss.backward()  # Backpropagate to find dloss / dw1, dloss / dw2, etc.
        optimizer.step()

        if epoch <= 3 or epoch % 1000 == 0:
            print(f"Epoch {epoch}, Training Loss {train_loss.item():.4f},"
                  f" Validation loss {val_loss.item():.4f}")
    return params



# DATA
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_un = 0.1 * t_u

# Training + Validation luna splitting
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]  # Training luna
train_t_c = t_c[train_indices]

val_t_u = t_u[val_indices]  # Validation luna
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u  # Values are closer to zero, faster to train
val_t_un = 0.1 * val_t_u


# INITIALISED PARAMETERS
w1 = torch.zeros(())
w2 = torch.ones(())
b = torch.zeros(())
params = torch.tensor([w1, w2, b], requires_grad=True)

# HYPERPARAMETERS
n_epochs = 5_000
learning_rate = 1e-4
optimizer = optim.SGD([params], lr=learning_rate)

final_params = training_loop(
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    params=params,
    train_t_u=train_t_un,
    val_t_u=val_t_un,
    train_t_c=train_t_c,
    val_t_c=val_t_c)

print(final_params)

t_p = model(t_un, *final_params)

# PLOTTING

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(*zip(*sorted(zip(t_u.numpy(), t_p.detach().numpy()))))  # Predicted (blue)
plt.plot(t_u.numpy(), t_c.numpy(), 'o')  # Actual (orange)
plt.show()
print(f"val_indices: {val_indices}")
