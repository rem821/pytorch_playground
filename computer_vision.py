import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
from helper_functions import accuracy_fn, print_train_time, train_step, test_step
from pathlib import Path


def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss,
            "model_acc": acc}


class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )

    def forward(self, x):
        return self.layers.forward(x)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    BATCH_SIZE = 32
    train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor(),
                                       target_transform=None)

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(),
                                      target_transform=None)

    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data loaders: {train_dataloader, test_dataloader}")
    print(f"Length of train_dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}...")
    print(f"Length of test_dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}...")

    class_names = train_data.classes
    print(class_names)

    image, label = train_data[0]

    print(f"Image shape: {image.shape} -> [color_channels, height, width]")
    print(f"Image label: {class_names[label]}")

    rows = 4
    cols = 4
    fig = plt.figure(figsize=(9, 9))
    for i in range(1, rows * cols + 1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        image, label = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(class_names[label])
        plt.axis(False)
    plt.show()

    model = FashionMNISTModelV0(input_shape=784, hidden_units=1024, output_shape=len(class_names))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    epochs = 100

    start_time = timer()
    for epoch in tqdm(range(epochs)):
        train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        test_step(model, test_dataloader, loss_fn, accuracy_fn, device)

    end_time = timer()
    print_train_time(start_time, end_time, device)

    model_results = eval_model(model.cpu(), test_dataloader, loss_fn, accuracy_fn)
    print(model_results)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "pytorch_computer_vision _model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
