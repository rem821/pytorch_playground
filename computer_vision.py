import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from helper_functions import *
from pathlib import Path
import random
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix


class FashionMNISTModel(nn.Module):
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


class FashionMNISTModelCnn(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


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

    rows, cols = 4, 4
    images = []
    labels = []
    for sample, label in random.sample(list(test_data), k=rows*cols):
        images.append(sample)
        labels.append(label)

    visualize_samples(rows, cols, images, class_names, labels)

    model = FashionMNISTModelCnn(input_shape=1, hidden_units=50, output_shape=len(class_names))
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    epochs = 10

    start_time = timer()
    for epoch in tqdm(range(epochs)):
        train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_step(model, test_dataloader, loss_fn, device)

    end_time = timer()
    print_train_time(start_time, end_time, device)

    model_results = eval_model(model.cpu(), test_dataloader, loss_fn, accuracy_fn)
    print(model_results)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "pytorch_computer_vision _model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    #test_samples = []
    #test_labels = []

    #for sample, label in random.sample(list(test_data), k=16):
    #    test_samples.append(sample)
    #    test_labels.append(label)

    #pred_probs = make_predictions(model, test_samples, device)
    #pred_labels = pred_probs.argmax(dim=1)
    #visualize_samples(rows, cols, test_samples, class_names, test_labels, pred_labels, True)

    test_samples = []
    test_labels = []
    for X_test, y_test in test_data:
        test_samples.append(X_test)
        test_labels.append(y_test)

    y_preds = make_predictions(model, test_samples, device).argmax(dim=1)
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_preds, target=test_data.targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10, 7),
    )
    fig.show()
