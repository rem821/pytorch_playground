import random

from torchvision import transforms
from torchvision.datasets import ImageFolder

from helper_functions import *
from torchinfo import summary
import os
from tqdm.auto import tqdm


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 13 * 13, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x)))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"CPU count: {os.cpu_count()}")

    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    if image_path.is_dir():
        print(f"{image_path} directory already exists")
    else:
        print("Downloading the dataset")
        image_path.mkdir(parents=True, exist_ok=True)

        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            requests = requests.get(
                "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            f.write(requests.content)

        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping...")
            zip_ref.extractall(image_path)

    walk_through_dir(image_path)

    train_path = image_path / "train"
    test_path = image_path / "test"

    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # Transform data
    train_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.ToTensor()
    ])

    # Create dataset
    train_dataset = ImageFolder(root=train_path, transform=train_transform, target_transform=None)
    test_dataset = ImageFolder(root=test_path, transform=test_transform, target_transform=None)

    print(train_dataset.classes)

    rows, cols = 4, 4
    samples = []
    labels = []
    for sample, label in random.sample(list(train_dataset), k=rows * cols):
        samples.append(sample.permute(1, 2, 0))
        labels.append(label)

    visualize_samples_rgb(rows, cols, samples, train_dataset.classes, labels)

    # Create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, num_workers=os.cpu_count(), shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=os.cpu_count(), shuffle=False)

    img, lbl = next(iter(train_dataloader))
    print(f"Image shape {img.shape}")
    print(f"Label shape: {lbl.shape}")

    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_dataset.classes)).to(device)

    image_batch, label_batch = next(iter(train_dataloader))
    summary(model, input_size=[32, 3, 64, 64])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    epochs = 1000
    results = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device)
    plot_loss_curves(results)

    model_results = eval_model(model.cpu(), test_dataloader, loss_fn, accuracy_fn)
    print(model_results)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "pytorch_custom_dataset_model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)
