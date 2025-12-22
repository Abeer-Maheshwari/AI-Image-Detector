# Importing all the required modules. If you run this code locally, make sure that you have installed all the requirements
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.01
WEIGHT_DECAY = 0.01
SEED = 676767
LOGARITHMIC_THRESHOLD = 10  # Loss value before scale becomes logarithmic.

# PATH for dataset (extracted folder name - "Human Faces Dataset" | Rename it to "human_faces_dataset")
# DATASET_PATH = "./human_faces_dataset"  # Google Colab
DATASET_PATH = './human_faces_dataset' # replace as necessary

# Saves the weights as a .pth in the same directory as the code.
WEIGHTS_PATH = Path(__file__).with_name('trained_models.pth')

# Makes everything deterministic(fair)
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Auto-detects GPU/MPS/CPU
# CUDA - GPU, MPS - Apple Silicon Chips
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
DEVICE = get_device()

# Simple CNN - 3 conv blocks, ReLU + MaxPool after each. Final fully-connected layers
class CNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Changes image size to 64x64 and randomly flips them
def build_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

# Loads the entire image dataset and returns a ready-to-use Pytorch dataset
def load_dataset(transform):
    return torchvision.datasets.ImageFolder(root=DATASET_PATH, transform=transform)

# Randomly splits the dataset into images - 80% train | 20% test
def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    return train_dataset, test_dataset

# Feeds data in batches to the model
def build_dataloaders(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

# Trains the CNN for 20 epochs with a given optimiser. Prints train/test loss every epoch. Returns fully trained model
def train_model(model, train_loader, test_loader, optimiser, criterion, epochs=EPOCHS):
    model.to(DEVICE)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return model


def clone_state_dict(state_dict, to_cpu=True):
    cloned = {}
    for k, v in state_dict.items():
        tensor = v.detach().clone()
        if to_cpu:
            tensor = tensor.cpu()
        cloned[k] = tensor
    return cloned

# Dictionary cotaining all 10 optimisers and their respective parameters
optimisers_dict = {
    'SGD': {
        'no_wd_fn': lambda params: optim.SGD(params, lr=LR),
        'wd_fn':    lambda params: optim.SGD(params, lr=LR, weight_decay=WEIGHT_DECAY),
    },
    'SGD_Momentum': {
        'no_wd_fn': lambda params: optim.SGD(params, lr=LR, momentum=0.9),
        'wd_fn':    lambda params: optim.SGD(params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY),
    },
    'RMSProp': {
        'no_wd_fn': lambda params: optim.RMSprop(params, lr=LR),
        'wd_fn':    lambda params: optim.RMSprop(params, lr=LR, weight_decay=WEIGHT_DECAY),
    },
    'Adagrad': {
        'no_wd_fn': lambda params: optim.Adagrad(params, lr=LR),
        'wd_fn':    lambda params: optim.Adagrad(params, lr=LR, weight_decay=WEIGHT_DECAY),
    },
    'Adadelta': {
        'no_wd_fn': lambda params: optim.Adadelta(params, lr=LR),
        'wd_fn':    lambda params: optim.Adadelta(params, lr=LR, weight_decay=WEIGHT_DECAY),
    },
    'Adam': {
        'no_wd_fn': lambda params: optim.Adam(params, lr=LR),
        'wd_fn':    lambda params: optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=False),
    },
    'AdamW': {
        # AdamW is inherently decoupled_weight_decay Adam; no `momentum` or `decoupled_weight_decay` args.
        'no_wd_fn': lambda params: optim.AdamW(params, lr=LR, weight_decay=0.0),
        'wd_fn':    lambda params: optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY),
    },
    'NAdam': {
        'no_wd_fn': lambda params: optim.NAdam(params, lr=LR),
        'wd_fn':    lambda params: optim.NAdam(params, lr=LR, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=False),
    },
    'NAdamW': {
        # NAdam with decoupled_weight_decay.
        'no_wd_fn': lambda params: optim.NAdam(params, lr=LR),
        'wd_fn':    lambda params: optim.NAdam(params, lr=LR, weight_decay=WEIGHT_DECAY, decoupled_weight_decay=True),
    },
    'Adamax': {
        'no_wd_fn': lambda params: optim.Adamax(params, lr=LR),
        'wd_fn':    lambda params: optim.Adamax(params, lr=LR, weight_decay=WEIGHT_DECAY),
    },
}

# Trains the optimisers twice, once with weight decay and once without. All start from the exact same initial weights and see the exact same data.
# Returns a dictionary containing every single trained model.
def train_all_optimisers(train_loader, test_loader, criterion):
    saved_states = {}
    for opt_name, opt_cfg in optimisers_dict.items():
        print(f"\n=== Running {opt_name} ===")
        model_no_wd = CNN().to(DEVICE)
        optimiser_no_wd = opt_cfg['no_wd_fn'](model_no_wd.parameters())
        initial_state = clone_state_dict(model_no_wd.state_dict(), to_cpu=True)

        print(f"Training {opt_name} without weight decay...")
        model_no_wd = train_model(model_no_wd, train_loader, test_loader, optimiser_no_wd, criterion)
        theta = clone_state_dict(model_no_wd.state_dict(), to_cpu=True)

        model_wd = CNN().to(DEVICE)
        model_wd.load_state_dict(initial_state)
        optimiser_wd = opt_cfg['wd_fn'](model_wd.parameters())
        print(f"Training {opt_name} with weight decay...")
        model_wd = train_model(model_wd, train_loader, test_loader, optimiser_wd, criterion)
        theta_prime = clone_state_dict(model_wd.state_dict(), to_cpu=True)

        saved_states[opt_name] = {
            'initial': initial_state,
            'no_wd': theta,
            'wd': theta_prime,
        }
    return saved_states

# Saves the trained model as a .pth file
def save_training_bundle(saved_states, train_dataset, test_dataset, save_path=WEIGHTS_PATH):
    payload = {
        'meta': {
            'img_size': IMG_SIZE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'seed': SEED,
            'logarithmic_threshold': LOGARITHMIC_THRESHOLD,
            'dataset_path': DATASET_PATH,
        },
        'splits': {
            'train_indices': list(train_dataset.indices) if isinstance(train_dataset, Subset) else None,
            'test_indices': list(test_dataset.indices) if isinstance(test_dataset, Subset) else None,
        },
        'optimisers': saved_states,
    }
    torch.save(payload, save_path)
    return save_path

# Main function that goes through the steps sequentially
def main():
    print(f"Using device: {DEVICE}")
    set_seed(SEED)
    start = time.time()
    transform = build_transform()
    dataset = load_dataset(transform)
    train_dataset, test_dataset = split_dataset(dataset)
    train_loader, test_loader = build_dataloaders(train_dataset, test_dataset)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    saved_states = train_all_optimisers(train_loader, test_loader, criterion)
    save_path = save_training_bundle(saved_states, train_dataset, test_dataset)
    elapsed_hours = (time.time() - start) / 60.0
    print(f"Training finished in {elapsed_hours:.3f} minutes. Saved weights to {save_path}.")

# Python's official way of running the main function when using OOP
if __name__ == '__main__':
    main()