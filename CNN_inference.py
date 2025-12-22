# Importing all the required modules. If you run this code locally, make sure that you have installed all the requirements
import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath,amssymb}'
    r'\boldmath'
    r'\bfseries'
)
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib.ticker import FuncFormatter
from torch.utils.data import DataLoader, Subset

ROOT_PARENT = Path(__file__).resolve().parent.parent

if str(ROOT_PARENT) not in sys.path:
    sys.path.append(str(ROOT_PARENT))

# Imports hyperparameters fron CNN_train file. Do not change the names of any files or the code will not function properly.
from CNN_train import (
    CNN,
    LOGARITHMIC_THRESHOLD,
    SEED,
    build_transform,
    set_seed,
)

WEIGHTS_PATH = Path(__file__).with_name('trained_models.pth') # PATH for the weights created by CNN_train. (Overriden later)
PLOTS_PATH = Path(__file__).resolve().with_name('loss_plots.png') # Saves the produced plot to the same directory as the code.
DEFAULT_ALPHA_POINTS = 100  # Number of α points to interpolate.

# Interpolation range (alpha range) for each optimiser.
ALPHA_SWEEP_CONFIG = {
    'SGD': {
        'init_vs_trained': {'range': (-0.5, 2.1)},
        'no_wd_vs_wd': {'range': (-2.5, 5)},
        'init_vs_trained_WD': {'range': (-0.5, 2.25)},
    },
    'SGD_Momentum': {
        'init_vs_trained': {'range': (-0.5, 3.75)},
        'no_wd_vs_wd': {'range': (-5, 5.5)},
        'init_vs_trained_WD': {'range': (-0.25, 1.5)},
    },
    'RMSProp': {
        'init_vs_trained': {'range': (-0.15, 5)},
        'no_wd_vs_wd': {'range': (-2.1, 1.5)},
        'init_vs_trained_WD': {'range': (-3, 5.5)},
    },
    'Adagrad': {
        'init_vs_trained': {'range': (-1.5, 3)},
        'no_wd_vs_wd': {'range': (-3, 3.5)},
        'init_vs_trained_WD': {'range': (-1, 1.75)},
    },
    'Adadelta': {
        'init_vs_trained': {'range': (-0.5, 1.75)},
        'no_wd_vs_wd': {'range': (-20, 12.5)},
        'init_vs_trained_WD': {'range': (-0.5, 1.7)},
    },
    'Adam': {
        'init_vs_trained': {'range': (-0.25, 5)},
        'no_wd_vs_wd': {'range': (-1, 2)},
        'init_vs_trained_WD': {'range': (-1.1, 5)},
    },
    'AdamW': {
        'init_vs_trained': {'range': (-0.2, 6)},
        'no_wd_vs_wd': {'range': (-3.25, 3)},
        'init_vs_trained_WD': {'range': (-0.3, 3)},
    },
    'NAdam': {
        'init_vs_trained': {'range': (-0.5, 5)},
        'no_wd_vs_wd': {'range': (-1.5, 1.75)},
        'init_vs_trained_WD': {'range': (-4, 6)},
    },
    'NAdamW': {
        'init_vs_trained': {'range': (-0.25, 6.25)},
        'no_wd_vs_wd': {'range': (-1.5, 1.75)},
        'init_vs_trained_WD': {'range': (-0.12, 5)},
    },
    'Adamax': {
        'init_vs_trained': {'range': (-1, 5)},
        'no_wd_vs_wd': {'range': (-4, 2)},
        'init_vs_trained_WD': {'range': (-1.5, 3)},
    },
}

# Auto-detects GPU/MPS/CPU
# CUDA - GPU, MPS - Apple Silicon Chips
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
device = get_device()

# Loads the .pth file that was created from CNN_train
def load_training_bundle(path=WEIGHTS_PATH, device=device):
    if not path.exists():
        raise FileNotFoundError(f"Missing weights file: {path}. Run CNN.py to train and save models first.")
    return torch.load(path, map_location=device)

# Recreates the exact same data loaders used during training by CNN_train.py
def build_eval_loaders(bundle):
    transform = build_transform()
    dataset = torchvision.datasets.ImageFolder(root=bundle['meta']['dataset_path'], transform=transform) # Uses path to the same dataset used in CNN_train. If you would like to change the path, change root="your dataset path here"
    train_indices = bundle['splits']['train_indices']
    test_indices = bundle['splits']['test_indices']
    if train_indices is None or test_indices is None:
        raise ValueError('Saved bundle is missing dataset split indices. Please re-run training to regenerate the file.')
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    batch_size = bundle['meta']['batch_size']
    eval_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    eval_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return eval_train_loader, eval_test_loader

# Returns 3 lists of alpha values (one per interpolation case)
# Each list is custom-tuned for that optimiser
# Ensures the loss curves are fully visible and comparable in the final plot
def _build_alpha_grid(opt_name, bundle):
    config = ALPHA_SWEEP_CONFIG.get(opt_name)
    if config is not None:
        alphas = {}
        for case, params in config.items():
            if 'range' not in params or len(params['range']) != 2:
                raise ValueError(f"Invalid alpha range specified for {opt_name}:{case}.")
            start, end = params['range']
            num_points = params.get('points', DEFAULT_ALPHA_POINTS)
            if num_points <= 0:
                raise ValueError(f"Alpha point count must be positive for {opt_name}:{case}.")
            alphas[case] = np.linspace(start, end, num_points)
        return alphas

    legacy_alphas = bundle.get('alphas', {}).get(opt_name)
    if legacy_alphas is not None:
        return {key: np.asarray(vals) for key, vals in legacy_alphas.items()}

    raise KeyError(
        f"Unable to build alpha sweep configuration for optimiser '{opt_name}'. "
        "Please update ALPHA_SWEEP_CONFIG or regenerate the training bundle."
    )


def _to_device_state_dict(state_dict, device):
    return {k: v.to(device) for k, v in state_dict.items()}

# builds model from previous file
def _build_eval_model(device):
    model = CNN().to(device)
    model.requires_grad_(False)
    return model


def _log_alpha_progress(idx, total, alpha, opt_name, case_label):
    print(f"Processing alpha {idx + 1}/{total}: {float(alpha):.4f} for {opt_name} [{case_label}]")

# Takes 2 sets of trained weights. Measures train and test loss at dozens of points along a straight line. Returns data for a 10x3 plot
def _run_interpolation_case(start_state, end_state, alphas, opt_name, case_label, eval_train_loader, eval_test_loader, criterion, device):
    model_eval = _build_eval_model(device)
    train_losses, test_losses = [], []
    total = len(alphas)
    for idx, alpha in enumerate(alphas):
        interpolate_parameters(model_eval, start_state, end_state, alpha)
        _log_alpha_progress(idx, total, alpha, opt_name, case_label)
        train_losses.append(evaluate_loss(model_eval, eval_train_loader, criterion, device))
        test_losses.append(evaluate_loss(model_eval, eval_test_loader, criterion, device))
    return train_losses, test_losses

# Sets the model to be a 50/50 blend of two trained CNNs
def interpolate_parameters(model, theta, theta_prime, alpha):
    with torch.inference_mode():
        new_state_dict = {}
        for k in theta:
            start = theta[k]
            end = theta_prime[k]
            alpha_tensor = start.new_tensor(float(alpha))
            new_state_dict[k] = torch.lerp(start, end, alpha_tensor)
        model.load_state_dict(new_state_dict, strict=True)
    return model

# Returns the true average loss of the model on the entire dataset
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

# Iterates between each optimiser for each interpolation case. Measures the train and test lost between the solutions.
def run_interpolation(bundle, device):
    criterion = nn.CrossEntropyLoss().to(device)
    eval_train_loader, eval_test_loader = build_eval_loaders(bundle)
    results = {}
    for opt_name, states in bundle['optimisers'].items():
        print(f"\nInterpolating {opt_name}...")
        alphas = _build_alpha_grid(opt_name, bundle)
        initial_state = _to_device_state_dict(states['initial'], device)
        theta = _to_device_state_dict(states['no_wd'], device)
        theta_prime = _to_device_state_dict(states['wd'], device)
        case_results = {}
        case_results['init_vs_trained'] = _run_interpolation_case(
            initial_state,
            theta,
            alphas['init_vs_trained'],
            opt_name,
            'init_vs_trained',
            eval_train_loader,
            eval_test_loader,
            criterion,
            device,
        )
        case_results['no_wd_vs_wd'] = _run_interpolation_case(
            theta,
            theta_prime,
            alphas['no_wd_vs_wd'],
            opt_name,
            'no_wd_vs_wd',
            eval_train_loader,
            eval_test_loader,
            criterion,
            device,
        )
        case_results['init_vs_trained_WD'] = _run_interpolation_case(
            initial_state,
            theta_prime,
            alphas['init_vs_trained_WD'],
            opt_name,
            'init_vs_trained_WD',
            eval_train_loader,
            eval_test_loader,
            criterion,
            device,
        )
        case_results['alphas'] = alphas
        results[opt_name] = case_results
    return results

# Takes interpolation data and plots the results in a 10x3 grid and saves an image to the disk
def plot_results(results):
    num_opts = len(results)
    plt.figure(figsize=(28, 5 * num_opts))  # nicer aspect ratio
    plt.suptitle("1-Dimensional Linear Interpolation in the Loss Landscape", fontsize=20, y=0.98)

    for i, (opt_name, data) in enumerate(results.items(), 1):
        alphas = data['alphas']
        cases = [
            ('init_vs_trained', 'Initial vs Trained (No WD)', 'tab:blue'),
            ('no_wd_vs_wd',     'No WD vs With WD',        'tab:orange'),
            ('init_vs_trained_WD', 'Initial vs Trained (WD)', 'tab:green'),
        ]

        for j, (case_key, title, color) in enumerate(cases, 1):
            ax = plt.subplot(num_opts, 3, 3*(i-1) + j)
            train_loss = data[case_key][0]
            test_loss  = data[case_key][1]
            alpha_vals = alphas[case_key]

            ax.plot(alpha_vals, train_loss, 'o-', color='blue', label='Train Loss', markersize=4)
            ax.plot(alpha_vals, test_loss,  's-', color='red',  label='Test Loss',  markersize=4)

            comb_train_test = np.concatenate((train_loss, test_loss))
            max_loss_val = np.max(comb_train_test)

            if max_loss_val >= 10:
                ax.set_yscale('log')

            ax.set_title(f"{opt_name}\n{title}", fontsize=12)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$L(\alpha \theta_1 + (1-\alpha)\theta_2)$")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            ax.legend()

    # Add extra padding so subplot titles and labels do not overlap
    plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    plt.savefig(PLOTS_PATH, dpi=400, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {PLOTS_PATH.resolve()}")

# Main function that goes through the steps sequentially
def main():
    set_seed(SEED)
    device = get_device()
    print(f'Using device: {device}')
    bundle = load_training_bundle(device=device)
    start = time.time()
    results = run_interpolation(bundle, device)
    elapsed_hours = (time.time() - start) / 60.0
    print(f'Interpolation finished in {elapsed_hours:.3f} minutes.')
    plot_results(results)

# Python's official way of running the main function when using OOP
if __name__ == '__main__':
    main()