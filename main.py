import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from helper_functions import plot_predictions


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02

    X = torch.arange(start, end, step).unsqueeze(1)
    y = X * weight + bias

    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model = LinearRegressionModel()
    model.to(device)
    print(model.state_dict())

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005)

    epochs = 2000

    epoch_no = []
    training_loss = []
    testing_loss = []

    for epoch in range(epochs):
        model.train()

        y_pred = model(X_train)

        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)

            test_loss = loss_fn(test_pred, y_test)

        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        epoch_no.append(epoch)
        training_loss.append(loss.cpu())
        testing_loss.append(test_loss.cpu())

    with torch.inference_mode():
        plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), test_pred.cpu())
        plt.plot(epoch_no, training_loss, label="Training loss")
        plt.plot(epoch_no, testing_loss, label="Testing loss")
        plt.show()

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "pytorch_workflow_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    # To load a model
    # model = LinearRegressionModel()
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
