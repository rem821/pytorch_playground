import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import plot_decision_boundary, accuracy_fn


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    n_samples = 1000

    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
    plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model = nn.Sequential(
        nn.Linear(in_features=2, out_features=10),
        nn.LeakyReLU(),
        nn.Linear(in_features=10, out_features=10),
        nn.LeakyReLU(),
        nn.Linear(in_features=10, out_features=1),
    )
    model = model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    epoch_no = []
    training_loss = []
    testing_loss = []
    accuracy = []
    test_accuracy = []

    epochs = 1000

    for epoch in range(epochs):
        model.train()

        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)

        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        test_acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        print(f"Epoch: {epoch} | Loss: {loss} | Acc: {acc} | Test loss: {test_loss} | Test Acc: {test_acc}")
        epoch_no.append(epoch)
        training_loss.append(loss.cpu())
        testing_loss.append(test_loss.cpu())
        accuracy.append(acc)
        test_accuracy.append(test_acc)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()
